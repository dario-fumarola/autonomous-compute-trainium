"""Pinned upstream FP32 check of the full captured next-TRAIN gradient."""
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
from compute_trainium.low_depth_gradient_probe import validate_probe_state
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_training_candidate import gradient_comparison,initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--point',required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--seconds-limit',type=int,default=1200);a=p.parse_args();started=time.time()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(12);torch.set_num_interop_threads(1)
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Gradient-probe plan changed')
    probe=json.loads(a.plan.read_text());point=probe['points'][a.point]
    if any(file_sha256(f)!=sha for f,sha in probe['source_sha256'].items()):raise AssertionError('Frozen gradient-check source changed')
    if subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean pinned upstream CPU reference required')
    sys.path.insert(0,str(a.upstream));parent_file=a.root/probe['parent_plan']
    if file_sha256(parent_file)!=probe['parent_plan_sha256']:raise AssertionError('Parent plan changed')
    parent=json.loads(parent_file.read_text());native=json.loads((a.capture/'report.json').read_text())
    if (native['status']!='CAPTURED_ZERO_UPDATE_LOW_DEPTH_GLOBAL_GRADIENT' or native['point']!=a.point
        or native['probe_plan_sha256']!=a.plan_sha or native['checkpoint_sha256']!=point['checkpoint_sha256']):
        raise ValueError('Capture identity differs')
    if file_sha256(a.capture/'gradient.pt')!=native['gradient_sha256'] or file_sha256(a.root/point['checkpoint'])!=point['checkpoint_sha256']:
        raise AssertionError('Gradient/checkpoint changed')
    saved=torch.load(a.root/point['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    case=validate_probe_state(saved,probe['parent_plan_sha256'],parent,point['case_id'])
    full_digest=digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})
    if full_digest!=saved['model_optimizer_digest'] or full_digest!=native['full_state_digest']:
        raise AssertionError('Independent complete-state readback differs')
    config=ParcaeCandidateConfig(**parent['model']);recurrent=case['arm']!='ordinary'
    model=make_reference(config,recurrent);model.load_state_dict(saved['model'],strict=True);del saved
    before=digest(dict(sorted(model.state_dict().items())))
    packed=PackedCorpus.from_mapped(a.root/parent['data']['packed'],1024,23,parent['data']['corpus_sha256'])
    cursor=native['source_cursor'];total=sum(packed.counts(cursor,native['windows']))
    if total!=native['real_targets']:raise AssertionError('Independent global source denominator differs')
    noise={}
    if recurrent:
        for rank,sha in enumerate(native['noise_sha256_by_rank']):
            path=a.capture/f'probe-h0-rank{rank}.npy'
            if file_sha256(path)!=sha:raise AssertionError('Captured hidden states changed')
            noise[rank]=np.load(path,mmap_mode='r')
    depth_file=parent_file.parent/case['depth_table']
    if file_sha256(depth_file)!=case['depth_table_sha256']:raise AssertionError('Independent depth table changed')
    depths=np.load(depth_file,mmap_mode='r')
    current=None
    if recurrent:model.initialize_state=lambda _:current
    loss=0.
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3 or min(shutil.disk_usage(a.output).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Independent gradient-check reserve')
        if time.time()-started>a.seconds_limit:raise TimeoutError('Independent gradient-check cap')
    with sdpa_kernel(SDPBackend.MATH):
        for index,row in enumerate(native['probe_rows']):
            guard();window=cursor+index
            if row['window']!=window:raise AssertionError('Global objective order differs')
            expected_window=routed_indices(cursor,row['micro'],row['rank'],32,1,parent['accumulation'],parent['routing_seed'])[0]
            expected_pair=tuple(map(int,depths[native['training_update'],row['micro'],row['rank']]))
            if expected_window!=window or expected_pair!=(row['n'],row['k']):
                raise AssertionError('Independent registered source/depth assignment differs')
            current=torch.from_numpy(np.array(noise[row['rank']][row['micro']:row['micro']+1],copy=True)) if recurrent else None
            if recurrent and digest(current)!=row['h0_digest']:raise AssertionError('Independent initial hidden state differs')
            batch=packed.batch([window],0)[0]
            kwargs={'num_steps_pair':torch.tensor([row['n']-row['k'],row['k']])} if recurrent else {}
            logits=model(batch.inputs,return_logits=True,**kwargs)['logits']
            value=(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none')*batch.target_mask.flatten()).sum()
            (value/total).backward();loss+=float(value.detach())
            if (index+1)%32==0:write_state(a.output/'progress.json',{'windows':index+1,'wall_seconds':time.time()-started})
    expected={n:p.grad.detach() for n,p in model.named_parameters()}
    actual=torch.load(a.capture/'gradient.pt',map_location='cpu',mmap=True,weights_only=False)
    comparison=gradient_comparison(expected,actual['gradients']);difference=actual['loss']-loss/total
    if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('Independent probe changed weights')
    passed=comparison['passes_two_percent_gate'] and abs(difference)<=.003
    result={'status':'PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT' if passed else 'FAIL_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT',
        'point':a.point,'case_id':point['case_id'],'arm':case['arm'],'training_update':native['training_update'],
        'gradient':comparison,'cpu_nll':loss/total,'native_minus_cpu_nll':difference,
        'real_targets':total,'windows':native['windows'],'checkpoint_sha256':point['checkpoint_sha256'],
        'full_state_readback_exact':True,'model_unchanged':True,'assigned_chips':0,'optimizer_updates':0,
        'research_optimizer_updates':0,'test_accessed':False,'wall_seconds':time.time()-started,
        'scope':'Independent pinned-upstream FP32 complete TRAIN gradient at frozen checkpoint. Exact state/depth/noise/source coverage, unchanged2% overall and family gates and0.003-nat loss gate. No held-out learning claim.'}
    write_state(a.output/'report.json',result);print(json.dumps(result),flush=True)
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
