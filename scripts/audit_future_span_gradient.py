"""Independent dense-target FP32 objective at a fitted scientific checkpoint."""
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
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.depth_mixture import routed_indices
from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_gradient_point import point_tables,validate_point
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,file_sha256
from scripts.audit_future_span_qualification import construct_reference,gradient_check
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--case',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha',required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--seconds-limit',type=int,default=1200);a=p.parse_args();started=time.time()
    a.output.mkdir(parents=True,exist_ok=False);result={'status':'STOPPED_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'}
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (available<544*1024**3 or min(shutil.disk_usage(a.output).free,shutil.disk_usage('/').free)<35*1024**3
                or time.time()-started>a.seconds_limit):raise RuntimeError('Independent mature-objective audit reserve/cap')
    try:
        guard();torch.set_num_threads(12);torch.set_num_interop_threads(1)
        if file_sha256(a.plan)!=a.plan_sha or file_sha256(a.checkpoint)!=a.checkpoint_sha:
            raise AssertionError('Frozen objective plan/checkpoint changed')
        plan=json.loads(a.plan.read_text())
        if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Saved objective/probe source changed')
        if (subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
                or subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'],text=True).strip()):
            raise ValueError('Clean pinned independent upstream required')
        sys.path.insert(0,str(a.upstream));case=plan['cases'][a.case]
        native=json.loads((a.capture/'report.json').read_text())
        if (native['status']!='CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT' or native['plan_sha256']!=a.plan_sha
                or native['case_id']!=a.case or native['checkpoint_sha256']!=a.checkpoint_sha or native['optimizer_updates']!=0):
            raise ValueError('Exact zero-update native point identity required')
        if file_sha256(a.capture/'gradient.pt')!=native['gradient_sha256']:raise AssertionError('Captured global gradient changed')
        saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
        counts,anchors,depths,work=point_tables(a.root,a.plan,plan,case);update,cursor=saved['update'],saved['cursor']
        validate_point(saved,a.plan_sha,plan,a.case,counts[cursor]-counts[0],anchors[cursor],work[:update].sum())
        complete=digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})
        if complete!=saved['model_optimizer_digest'] or complete!=native['full_state_digest']:
            raise AssertionError('Independent complete-state readback differs')
        config=ParcaeCandidateConfig(**plan['model']);span=FutureSpanConfig(**plan['future_config'])
        model=construct_reference(config,saved['model']);before=digest(dict(sorted(model.state_dict().items())));del saved
        packed=PackedCorpus.from_mapped(a.root/plan['data']['packed'],1024,23,plan['data']['corpus_sha256'])
        if packed.fingerprint!=plan['data']['packing_fingerprint']:raise AssertionError('Independent natural packing differs')
        total=sum(packed.counts(cursor,512));anchor_count=0
        for window in range(cursor,cursor+512):
            base=window*1024
            anchor_count+=sum(bool((packed.domains[base+pos:base+pos+span.horizon+1]>0).all()) for pos in span.positions)
        if (cursor!=native['source_cursor'] or update!=native['training_update'] or native['windows']!=512
                or total!=native['real_targets'] or anchor_count!=native['valid_anchors']
                or len(native['rows'])!=512 or anchor_count!=int(anchors[cursor+512]-anchors[cursor])):
            raise AssertionError('Independent global source/anchor coverage differs')
        noise={}
        for rank,sha in enumerate(native['noise_sha256_by_rank']):
            path=a.capture/f'h0-rank{rank}.npy'
            if file_sha256(path)!=sha:raise AssertionError('Exact supplied latent states changed')
            noise[rank]=np.load(path,mmap_mode='r')
        if set(noise)!=set(range(32)):raise AssertionError('Incomplete latent-state replica capture')
        current=None;hidden={};model.initialize_state=lambda _:current
        hook=model.transformer.ln_f.register_forward_hook(lambda m,ins,out:hidden.update(value=out))
        ntp_sum=aux_sum=0.;matrix=0
        with sdpa_kernel(SDPBackend.MATH):
            for index,row in enumerate(native['rows']):
                guard();window=cursor+index
                if (row['window']!=window or routed_indices(cursor,row['micro'],row['rank'],32,1,16,plan['routing_seed'])[0]!=window
                        or tuple(map(int,depths[update,row['micro'],row['rank']]))!=(row['n'],row['k']) or row['n']!=row['k']):
                    raise AssertionError('Independent immutable depth/source/routing assignment differs')
                current=torch.from_numpy(np.array(noise[row['rank']][row['micro']:row['micro']+1],copy=True))
                if digest(current)!=row['h0_digest']:raise AssertionError('Independent exact latent state differs')
                batch=packed.batch([window],0)[0]
                logits=model(batch.inputs,return_logits=True,num_steps_pair=torch.tensor([0,row['n']]))['logits']
                future=model.future_lm_head(hidden['value'].index_select(1,model.future_positions))
                truth=torch.zeros_like(future);valid=[]
                for j,pos in enumerate(span.positions):
                    base=window*1024+pos
                    truth[0,j,list(set(map(int,packed.tokens[base+2:base+span.horizon+1])))]=1
                    valid.append(bool((packed.domains[base:base+span.horizon+1]>0).all()))
                if sum(valid)!=row['valid_anchors']:raise AssertionError('Independent per-window anchor count differs')
                mask=torch.tensor(valid,dtype=torch.bool)[None]
                auxiliary=(F.binary_cross_entropy_with_logits(future,truth,reduction='none').sum(-1)*mask).sum()/(span.horizon-1)
                ntp=(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none')*batch.target_mask.flatten()).sum()
                (ntp/total+case['objective_weight']*auxiliary/max(1,anchor_count)).backward()
                ntp_sum+=float(ntp.detach());aux_sum+=float(auxiliary.detach())
                matrix+=matmul_ledger(config,1024,total_steps=row['n'],backward_cap=row['n'])['training_matmul_flops']
                matrix+=6*len(span.positions)*config.dim*config.vocab_size
                if (index+1)%32==0:write_state(a.output/'progress.json',{'windows':index+1,'wall_seconds':time.time()-started})
        hook.remove();expected={n:p.grad.detach() for n,p in model.named_parameters()}
        if any(not torch.isfinite(g).all() for g in expected.values()):raise FloatingPointError('Nonfinite independent mature gradient')
        actual=torch.load(a.capture/'gradient.pt',map_location='cpu',mmap=True,weights_only=False)
        gradient=gradient_check(expected,actual['gradients'])
        differences={'ntp_nll':actual['ntp_nll']-ntp_sum/total,'future_mean':actual['future_mean']-aux_sum/max(1,anchor_count)}
        if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('Independent backward changed weights')
        if matrix!=native['diagnostic_matrix_ops']:raise AssertionError('Independent diagnostic work count differs')
        guard();passed=gradient['passes_two_percent_gate'] and all(abs(x)<=.003 for x in differences.values())
        result={'status':'PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT' if passed else 'FAIL_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT',
            'case_id':a.case,'cell':case['cell'],'training_update':update,'checkpoint_sha256':a.checkpoint_sha,
            'gradient':gradient,'loss_differences':differences,'cpu_ntp_nll':ntp_sum/total,'cpu_future_mean':aux_sum/max(1,anchor_count),
            'windows':512,'real_targets':total,'valid_anchors':anchor_count,'diagnostic_matrix_ops':matrix,
            'full_state_readback_exact':True,'model_unchanged':True}
        if not passed:raise RuntimeError('Unchanged2% body/family/head and0.003-nat loss gates failed')
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(assigned_chips=0,optimizer_updates=0,research_optimizer_updates=0,test_accessed=False,
            wall_seconds=time.time()-started,
            scope='Independent pinned-upstream FP32 full512-window natural TRAIN objective,dense independently constructed future targets. No optimizer step,held-out quality score or threshold change.')
        write_state(a.output/'report.json',result)


if __name__=='__main__':main()
