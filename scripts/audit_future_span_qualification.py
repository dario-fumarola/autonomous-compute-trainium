"""Independent FP32 full-objective and observed-gradient optimizer audits."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.depth_mixture import routed_indices
from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_reference_optimizer import reference_optimizer,fp32_reference_execution
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_recovery_check import moments
from compute_trainium.parcae_runtime_scalars import changing_qualification_schedule
from compute_trainium.parcae_training_candidate import gradient_comparison
from compute_trainium.parcae_update_qualification import relative_error,named_moments,compare_update,compare_moments
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.run_parcae_optimizer_native import step_comparison,step_passes


def gradient_check(expected,actual):
    body=lambda a:{k:v for k,v in a.items() if not k.startswith('future_lm_head.')}
    head=lambda a:{k:v for k,v in a.items() if k.startswith('future_lm_head.')}
    main=gradient_comparison(body(expected),body(actual));extra=relative_error(head(expected),head(actual))
    named={k:relative_error({k:expected[k]},{k:actual[k]}) for k in head(expected)}
    return {'body':main,'future_head':extra,'future_head_named':named,
        'passes_two_percent_gate':main['passes_two_percent_gate'] and extra['relative_l2']<=.02
        and all(v['relative_l2']<=.02 for v in named.values())}


def construct_reference(model_config,model_state):
    model=make_reference(model_config,True)
    model.load_state_dict({k:v for k,v in model_state.items() if not k.startswith('future_lm_head.') and k!='future_positions'},strict=True)
    model.add_module('future_lm_head',nn.Linear(model_config.dim,model_config.vocab_size,bias=True))
    model.register_buffer('future_positions',model_state['future_positions'].clone())
    model.load_state_dict(model_state,strict=True)
    return model


def audit():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--update',type=int,choices=[0,120],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--reuse-gold',type=Path);parser.add_argument('--seconds-limit',type=int,default=1200)
    args=parser.parse_args();started=time.time();args.output.mkdir(parents=True,exist_ok=False)
    result={'status':'STOPPED_INDEPENDENT_FUTURE_SPAN_AUDIT'};cpu_steps=0

    def guard():
        if time.time()-started>args.seconds_limit:raise TimeoutError('Original future audit CPU cap')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3 or min(shutil.disk_usage(args.output).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Independent future audit resource reserve')

    try:
        guard();torch.set_num_threads(12);torch.set_num_interop_threads(1)
        if file_sha256(args.plan)!=args.plan_sha:raise AssertionError('Frozen future qualification plan changed')
        plan=json.loads(args.plan.read_text())
        if any(file_sha256(p)!=sha for p,sha in plan['source_sha256'].items()):raise AssertionError('Frozen future qualification source changed')
        if (subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
                or subprocess.check_output(['git','-C',str(args.upstream),'status','--porcelain'],text=True).strip()):
            raise ValueError('Clean pinned independent upstream required')
        sys.path.insert(0,str(args.upstream))
        preparation=args.root/plan['preparation']
        if file_sha256(preparation/'manifest.json')!=plan['preparation_manifest_sha256']:raise AssertionError('CPU preparation changed')
        manifest=json.loads((preparation/'manifest.json').read_text());model_config=ParcaeCandidateConfig(**manifest['model'])
        span=FutureSpanConfig(**manifest['future_config']);native=json.loads((args.capture/'report.json').read_text())
        if native['status']!='CAPTURED_FUTURE_SPAN_TRAINING_QUALIFICATION' or native['signature']['plan_sha256']!=args.plan_sha:
            raise ValueError('Completed same-plan native qualification required')
        case=native['case'];point=next(p for p in native['probes'] if p['update']==args.update)
        if plan['cases'][case['identity']]!=case:raise AssertionError('Native case identity differs')
        if file_sha256(args.capture/point['gradient_file'])!=point['gradient_sha256']:raise AssertionError('Captured global vector changed')
        vector=torch.load(args.capture/point['gradient_file'],map_location='cpu',mmap=True,weights_only=False)
        state_file=preparation/manifest['initial_file'] if args.update==0 else args.capture/'full-state-u120.pt'
        expected_sha=manifest['initial_sha256'] if args.update==0 else native['full_checkpoint_sha256']
        if file_sha256(state_file)!=expected_sha:raise AssertionError('Exact audit state changed')
        state=torch.load(state_file,map_location='cpu',mmap=True,weights_only=False)
        model_digest=digest(dict(sorted(state['model'].items())))
        if args.update==120 and digest({'model':dict(sorted(state['model'].items())),'optimizer':state['optimizer']})!=native['full_checkpoint_state_digest']:
            raise AssertionError('Full optimizer/model state readback differs')
        packed=PackedCorpus.from_mapped(args.root/manifest['data']['packed'],1024,23,manifest['data']['corpus_sha256'])
        cursor=args.update*512;total=sum(packed.counts(cursor,512))
        if cursor!=point['cursor'] or total!=point['real_targets']:raise AssertionError('Independent global real-token denominator differs')
        d=manifest['depths'][case['arm']]
        if file_sha256(preparation/d['file'])!=d['sha256']:raise AssertionError('Independent full-gradient depth table changed')
        depths=np.load(preparation/d['file'],mmap_mode='r')
        noise={}
        for rank,sha in enumerate(point['noise_sha256_by_rank']):
            file=args.capture/f'h0-u{args.update}-rank{rank}.npy'
            if file_sha256(file)!=sha:raise AssertionError('Exact supplied initial states changed')
            noise[rank]=np.load(file,mmap_mode='r')
        if len(point['rows'])!=512:raise AssertionError('All512 original windows are required')
        for i,row in enumerate(point['rows']):
            if (row['window']!=cursor+i or routed_indices(cursor,row['micro'],row['rank'],32,1,16,manifest['routing_seed'])[0]!=row['window']
                    or tuple(map(int,depths[args.update,row['micro'],row['rank']]))!=(row['n'],row['k']) or row['n']!=row['k']):
                raise AssertionError('Independent source/depth/routing policy differs')
        anchor_count=0
        for window in range(cursor,cursor+512):
            base=window*1024;real=packed.domains[base:base+1025]
            anchor_count+=sum(bool((real[pos:pos+span.horizon+1]>0).all()) for pos in span.positions)
        if anchor_count!=point['valid_anchors']:raise AssertionError('Independent global future-anchor denominator differs')
        fingerprint={'case':case,'update':args.update,'model_digest':model_digest,'source_cursor':cursor,
            'real_targets':total,'valid_anchors':anchor_count,'noise_sha256_by_rank':point['noise_sha256_by_rank'],
            'depth_table_sha256':d['sha256'],'plan_sha256':args.plan_sha}
        model=None;matrix_work=0
        if args.reuse_gold:
            previous=json.loads((args.reuse_gold/'report.json').read_text())
            if (previous['status']!='PASS_INDEPENDENT_FUTURE_SPAN_QUALIFICATION_AUDIT'
                    or previous['gold_fingerprint']!=fingerprint or previous['reused_gold']):
                raise ValueError('Only independently computed gold at the identical state/source/noise/depth may be reused')
            gold_file=args.reuse_gold/'cpu-global.pt'
            if file_sha256(gold_file)!=previous['cpu_global_sha256']:raise AssertionError('Independent retained gold changed')
            gold=torch.load(gold_file,map_location='cpu',mmap=True,weights_only=False)
            expected=gold['gradients'];ntp_loss=gold['ntp_nll'];future_loss=gold['future_mean']
        else:
            model=construct_reference(model_config,state['model']);current=None;hidden={}
            model.initialize_state=lambda _:current
            hook=model.transformer.ln_f.register_forward_hook(lambda m,ins,out:hidden.update(value=out))
            ntp_sum=aux_sum=0.
            with sdpa_kernel(SDPBackend.MATH):
                for i,row in enumerate(point['rows']):
                    guard();window=row['window'];batch=packed.batch([window],0)[0]
                    current=torch.from_numpy(np.array(noise[row['rank']][row['micro']:row['micro']+1],copy=True))
                    if digest(current)!=row['h0_digest']:raise AssertionError('Exact CPU initial state differs')
                    logits=model(batch.inputs,return_logits=True,num_steps_pair=torch.tensor([0,row['n']]))['logits']
                    aux=model.future_lm_head(hidden['value'].index_select(1,model.future_positions))
                    truth=torch.zeros_like(aux);valid=[]
                    for j,pos in enumerate(span.positions):
                        base=window*1024+pos
                        ids=set(map(int,packed.tokens[base+2:base+span.horizon+1]));truth[0,j,list(ids)]=1
                        valid.append(bool((packed.domains[base:base+span.horizon+1]>0).all()))
                    mask=torch.tensor(valid,dtype=torch.bool)[None]
                    aux_value=(F.binary_cross_entropy_with_logits(aux,truth,reduction='none').sum(-1)*mask).sum()/(span.horizon-1)
                    ntp=(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none')*batch.target_mask.flatten()).sum()
                    (ntp/total+case['objective_weight']*aux_value/max(1,anchor_count)).backward()
                    ntp_sum+=float(ntp.detach());aux_sum+=float(aux_value.detach())
                    matrix_work+=matmul_ledger(model_config,1024,total_steps=row['n'],backward_cap=row['n'])['training_matmul_flops']
                    matrix_work+=6*len(span.positions)*model_config.dim*model_config.vocab_size
                    if (i+1)%32==0:write_state(args.output/'progress.json',{'windows':i+1,'wall_seconds':time.time()-started})
            hook.remove();expected={name:p.grad.detach().clone() for name,p in model.named_parameters()}
            if digest(dict(sorted(model.state_dict().items())))!=model_digest:raise AssertionError('Independent CPU backward changed weights')
            ntp_loss,future_loss=ntp_sum/total,aux_sum/max(1,anchor_count)
            gold={'gradients':expected,'ntp_nll':ntp_loss,'future_mean':future_loss,'fingerprint':fingerprint}
        if any(not torch.isfinite(g).all() for g in expected.values()):raise FloatingPointError('Nonfinite independent objective gradient')
        gradient=gradient_check(expected,vector['gradients'])
        differences={'ntp_nll':vector['ntp_nll']-ntp_loss,'future_mean':vector['future_mean']-future_loss}
        loss_ok=all(abs(x)<=.003 for x in differences.values())
        atomic_checkpoint(args.output/'cpu-global.pt',gold)
        # Isolate clipping/optimizer arithmetic using the observed native raw
        # gradient. Neural-gradient fidelity was independently checked above.
        if model is None:model=construct_reference(model_config,state['model'])
        for name,p in model.named_parameters():p.grad=vector['gradients'][name].clone()
        actual_file=args.capture/('after-first-update.pt' if args.update==0 else 'after121.pt')
        actual_sha=native['first_update_sha256'] if args.update==0 else native['final_sha256']
        if file_sha256(actual_file)!=actual_sha:raise AssertionError('Actual optimizer update changed')
        actual=torch.load(actual_file,map_location='cpu',mmap=True,weights_only=False)
        grouping=actual['optimizer']['recipe']['groups']
        optimizer=reference_optimizer(model,grouping,state.get('optimizer'))
        before={name:p.detach().clone() for name,p in model.named_parameters()}
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False))
        changing_qualification_schedule(optimizer,args.update)
        with fp32_reference_execution():optimizer.step()
        cpu_steps+=1
        expected_step={'model':{name:p.detach() for name,p in model.named_parameters()},
            'moments':named_moments(model,optimizer,reference=True)}
        native_step={'model':{name:actual['model'][name] for name in before},'moments':moments(actual['optimizer'])}
        step=step_comparison(before,expected_step,before,native_step)
        head_names=[name for name in before if name.startswith('future_lm_head.')]
        step['future_head']={'update_comparison':compare_update(before,expected_step['model'],before,native_step['model'],names=head_names),
            'moment_comparison':compare_moments({n:expected_step['moments'][n] for n in head_names},{n:native_step['moments'][n] for n in head_names})}
        step_ok=step_passes(step) and all(v['relative_l2']<=.02 for v in step['future_head'].values())
        guard();passed=gradient['passes_two_percent_gate'] and loss_ok and step_ok
        result={'status':'PASS_INDEPENDENT_FUTURE_SPAN_QUALIFICATION_AUDIT' if passed else 'FAIL_INDEPENDENT_FUTURE_SPAN_QUALIFICATION_AUDIT',
            'case':case,'update':args.update,'native_mode':native['mode'],'gradient':gradient,'loss_differences':differences,
            'cpu_ntp_nll':ntp_loss,'cpu_future_mean':future_loss,'first_or_recovered_update':step,
            'optimizer_step_passes_two_percent_gate':step_ok,'cpu_raw_gradient_norm_before_clip':norm,
            'gold_fingerprint':fingerprint,'reused_gold':args.reuse_gold is not None,
            'cpu_global_sha256':file_sha256(args.output/'cpu-global.pt'),'diagnostic_matrix_operations':matrix_work}
        if not passed:raise RuntimeError('Unchanged full-gradient/loss/update qualification thresholds failed')
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(assigned_chips=0,optimizer_updates=0,prescribed_cpu_reference_optimizer_steps=cpu_steps,
            research_optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,
            scope='Independent pinned upstream FP32 full512-window natural TRAIN objective with separately constructed dense presence targets. Exact state/source/depth/noise required for gold reuse. Native raw-gradient replay isolates clipping and independent optimizer arithmetic;no held-out score or scientific training adoption.')
        write_state(args.output/'report.json',result)


if __name__=='__main__':audit()
