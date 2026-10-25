"""Bounded TRAIN-only future-span head/gradient feasibility, no optimizer steps."""
import argparse
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from compute_trainium.future_span import (
    FutureSpanCandidate,FutureSpanConfig,future_span_targets,
    future_span_loss_sum,normalized_objective,
)
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_training_candidate import initial_state,gradient_comparison
from compute_trainium.parcae_update_qualification import relative_error
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();started=time.time()
    out=args.root/'future-span-cpu-screen-v1';public=Path('results/future_span_cpu_20261025')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    main_file=args.root/'parcae-low-depth-pilot-v1/main-plan-recovery-v2.json'
    parent=json.loads(main_file.read_text());model_config=ParcaeCandidateConfig(**parent['model'])
    config=FutureSpanConfig();cursor=576*512;windows=list(range(cursor,cursor+8))
    points=[]
    for arm,depth in [('fixed1',1),('poisson2',2)]:
        report=json.loads((args.root/f'runs/low-depth-main-r2-{arm}-first/report.json').read_text())
        point=next(row for row in report['checkpoint_records'] if row['update']==576)
        points.append({'arm':arm,'depth':depth,'backward_cap':depth,
            'checkpoint':f'runs/low-depth-main-r2-{arm}-first/checkpoint-u576.pt',
            'checkpoint_sha256':point['checkpoint_sha256'],'full_state_digest':point['full_state_digest'],
            'case_id':report['case_id']})
    sources={**parent['source_sha256'],
        'compute_trainium/future_span.py':file_sha256('compute_trainium/future_span.py'),
        'scripts/screen_future_span_cpu.py':file_sha256(__file__)}
    plan={'identity':'future-span-cpu-screen-v1','registered_unix':started,
        'main_plan_sha256':file_sha256(main_file),'points':points,
        'future_config':asdict(config),'source_windows':windows,'source_sha256':sources,
        'upstream_commit':PIN,'seconds_limit':900,'gradient_relative_gate':.02,
        'loss_absolute_gate':.003,'head_seed':104729,'assigned_chips':0,'optimizer_updates':0,
        'test_accessed':False,'scope':'Small CPU numerical preparation on identical eight natural TRAIN '
        'windows at immutable checkpoints. Independent pinned upstream FP32 backbone plus independent '
        'softplus presence loss versus FP16 candidate,FP32 auxiliary/next-token heads and scale8192. '
        'Full gradients at fixed supported depth1/2. No held-out scoring,updates,native qualification '
        'or scientific evidence about a future-objective gain.'}
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    records=[];comparisons=[];diagnostic_work=0;result={'status':'STOPPED_FUTURE_SPAN_CPU_SCREEN'}

    def guard():
        if time.time()-started>plan['seconds_limit']:raise TimeoutError('Original CPU feasibility cap')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('CPU feasibility memory/disk reserve')

    try:
        guard();torch.set_num_threads(8);torch.set_num_interop_threads(1)
        upstream=args.root/'upstream-parcae-69284c1'
        if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
                or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):
            raise ValueError('Clean pinned independent upstream required')
        sys.path.insert(0,str(upstream))
        packed=PackedCorpus.from_mapped(args.root/parent['data']['packed'],1024,23,parent['data']['corpus_sha256'])
        total=sum(packed.counts(cursor,len(windows)))
        batches=[]
        for window in windows:
            source=packed.tokens[window*1024:window*1024+1025].long()[None]
            real=packed.domains[window*1024:window*1024+1025][None]>0
            labels=future_span_targets(source,real,config,model_config.vocab_size)
            batches.append((window,packed.batch([window],0)[0],labels))
        anchors=sum(int(t.valid.sum()) for _,_,t in batches)
        if not anchors:raise ValueError('No complete same-document future spans in selected TRAIN windows')
        for point in points:
            guard();checkpoint=args.root/point['checkpoint']
            if file_sha256(checkpoint)!=point['checkpoint_sha256']:raise AssertionError('Training checkpoint changed')
            saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
            if digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})!=point['full_state_digest']:
                raise AssertionError('Complete predecessor state readback differs')
            case=parent['cases'][point['case_id']];vectors={};losses={};head_state=None
            for mixed in [False,True]:
                guard();begin=time.time();label='fp16' if mixed else 'independent_fp32'
                candidate=FutureSpanCandidate(model_config,config,head_seed=plan['head_seed'],initialize=False,fp16_body=mixed)
                missing=candidate.load_state_dict(saved['model'],strict=False)
                if set(missing.missing_keys)!={'future_lm_head.weight','future_lm_head.bias','future_positions'} or missing.unexpected_keys:
                    raise AssertionError('Auxiliary head changed the backbone inventory')
                if digest(dict(sorted(candidate.prediction_state_dict().items())))!=digest(dict(sorted(saved['model'].items()))):
                    raise AssertionError('Backbone changed while adding auxiliary head')
                if mixed:model=candidate
                else:
                    model=make_reference(model_config,True);model.load_state_dict(saved['model'],strict=True)
                    model.add_module('future_lm_head',copy.deepcopy(candidate.future_lm_head))
                    head_state={k:v.clone() for k,v in model.future_lm_head.state_dict().items()}
                    del candidate
                if any(not torch.equal(v,head_state[k]) for k,v in model.future_lm_head.state_dict().items()):
                    raise AssertionError('Precision comparison changed auxiliary head initialization')
                before=digest(dict(sorted(model.state_dict().items())));scale=8192 if mixed else 1
                ntp_sum=aux_sum=0.;current=None;hidden={}
                if not mixed:
                    model.initialize_state=lambda _:current.clone()
                    hook=model.transformer.ln_f.register_forward_hook(lambda m,ins,out:hidden.update(value=out))
                with sdpa_kernel(SDPBackend.MATH),FlopCounterMode(display=False) as counter:
                    for window,batch,targets in batches:
                        guard();current=initial_state(model_config,window,case['state_seed'])
                        n=point['depth']
                        if mixed:
                            logits,aux=model(batch.inputs,current,total_steps=n,backward_cap=n)
                            future=future_span_loss_sum(aux,targets,config)
                        else:
                            logits=model(batch.inputs,return_logits=True,num_steps_pair=torch.tensor([0,n]))['logits']
                            aux=model.future_lm_head(hidden['value'].index_select(1,targets.positions))
                            # Independent target implementation and closed-form BCE.
                            truth=torch.zeros_like(aux)
                            for j,pos in enumerate(config.positions):
                                ids=set(map(int,packed.tokens[window*1024+pos+2:window*1024+pos+config.horizon+1]))
                                truth[0,j,list(ids)]=1
                            future=((F.softplus(aux)-truth*aux).sum(-1)*targets.valid).sum()/(config.horizon-1)
                        ntp=(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none')*batch.target_mask.flatten()).sum()
                        value=normalized_objective(ntp,future,ntp_count=total,future_count=anchors,weight=config.weight)
                        (value*scale).backward();ntp_sum+=float(ntp.detach());aux_sum+=float(future.detach())
                if not mixed:hook.remove()
                vector={name:p.grad.detach().clone()/scale for name,p in model.named_parameters()}
                if any(not torch.isfinite(g).all() for g in vector.values()):raise ValueError('Nonfinite combined gradient')
                if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('Zero-update feasibility changed weights')
                expected=len(windows)*(matmul_ledger(model_config,1024,total_steps=point['depth'],
                    backward_cap=point['depth'])['training_matmul_flops']+6*len(config.positions)*model_config.dim*model_config.vocab_size)
                if counter.get_total_flops()!=expected:raise AssertionError('Complete auxiliary matrix-work count differs')
                diagnostic_work+=expected;vectors[label]=vector;losses[label]=ntp_sum/total+config.weight*aux_sum/anchors
                records.append({'arm':point['arm'],'depth':point['depth'],'precision':label,
                    'ntp_nll':ntp_sum/total,'future_loss_per_valid_anchor':aux_sum/anchors,
                    'combined_objective':losses[label],'auxiliary_parameters':sum(p.numel() for p in model.future_lm_head.parameters()),
                    'executed_matrix_operations':expected,'all_weights_unchanged':True,'wall_seconds':time.time()-begin})
                write_state(public/'progress.json',{'records':records})
                del model,vector;gc.collect()
            body={k:v for k,v in vectors['independent_fp32'].items() if not k.startswith('future_lm_head.')}
            actual_body={k:v for k,v in vectors['fp16'].items() if not k.startswith('future_lm_head.')}
            body_comparison=gradient_comparison(body,actual_body)
            head_comparison=relative_error({k:v for k,v in vectors['independent_fp32'].items() if k.startswith('future_lm_head.')},
                                          {k:v for k,v in vectors['fp16'].items() if k.startswith('future_lm_head.')})
            difference=losses['fp16']-losses['independent_fp32']
            comparisons.append({'arm':point['arm'],'body_gradient':body_comparison,'future_head_gradient':head_comparison,
                'combined_loss_difference':difference,'passes_small_cpu_screen':body_comparison['passes_two_percent_gate']
                and head_comparison['relative_l2']<=.02 and abs(difference)<=.003})
            del vectors,saved;gc.collect()
        if any(file_sha256(p)!=sha for p,sha in sources.items()):raise AssertionError('Frozen CPU feasibility source changed')
        result={'status':'PASS_SMALL_CPU_FUTURE_SPAN_SCREEN' if all(c['passes_small_cpu_screen'] for c in comparisons)
                else 'FAIL_SMALL_CPU_FUTURE_SPAN_SCREEN','comparisons':comparisons,
            'unique_train_windows':len(windows),'unique_real_targets':total,'valid_anchors':anchors,
            'executed_anchors_per_forward':len(windows)*len(config.positions),
            'diagnostic_real_target_presentations':4*total,'diagnostic_matrix_operations':diagnostic_work,
            'native_qualification_established':False,'scientific_followup_launched':False}
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(records=records,wall_seconds=time.time()-started,assigned_chips=0,optimizer_updates=0,
            test_accessed=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}),flush=True)


if __name__=='__main__':main()
