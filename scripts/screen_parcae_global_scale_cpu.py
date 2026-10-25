"""Loss-scale sensitivity using the actual512-window global normalizer.

An eight-window mean gives much larger intermediate gradients than each
microbatch receives in the real512-window objective. This screen retains
that original global denominator while computing only eight contributions.
It is not a replacement for a complete native/global gradient check.
"""
import argparse
import gc
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

from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_fp16_candidate import ParcaeFP16Candidate
from compute_trainium.parcae_training_candidate import gradient_comparison
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    probe=a.root/'low-depth-main-r2-gradient-poisson2-matched';probe_file=probe/'plan.json'
    probe_plan=json.loads(probe_file.read_text());point=probe_plan['points']['poisson2']
    capture=a.root/'runs/low-depth-main-r2-gradient-poisson2-matched-poisson2'
    native=json.loads((capture/'report.json').read_text());parent_file=a.root/probe_plan['parent_plan']
    parent=json.loads(parent_file.read_text());checkpoint=a.root/point['checkpoint']
    if (native['status']!='CAPTURED_ZERO_UPDATE_LOW_DEPTH_GLOBAL_GRADIENT'
            or native['probe_plan_sha256']!=file_sha256(probe_file)
            or file_sha256(parent_file)!=probe_plan['parent_plan_sha256']
            or file_sha256(checkpoint)!=point['checkpoint_sha256']):raise ValueError('Exact completed same-state global probe required')
    out=a.root/'parcae-global-scale-cpu-v1';public=Path('results/parcae_global_scale_cpu_20261025')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    source={**probe_plan['source_sha256'],'scripts/screen_parcae_global_scale_cpu.py':file_sha256(__file__)}
    plan={'identity':'parcae-global-scale-cpu-v1','registered_unix':started,'source_sha256':source,
        'probe_plan_sha256':file_sha256(probe_file),'capture_report_sha256':file_sha256(capture/'report.json'),
        'checkpoint_sha256':point['checkpoint_sha256'],'source_cursor':native['source_cursor'],
        'partial_windows':8,'global_windows':512,'global_real_token_denominator':native['real_targets'],
        'loss_scales':[8192,65536,524288],'seconds_limit':300,'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,
        'scope':'CPU numerical sensitivity of first8 exact natural TRAIN/source/depth/noise contributions at the mature recurrent checkpoint. Uses original512-window global denominator for every backward. FP32 reference versus FP16 body/FP32 head at three fixed manual scales. Native operators,flush-to-zero behavior,full512-vector fidelity,updates,recovery and speed are not established. No current/future frozen execution changes.'}
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    result={'status':'STOPPED_CPU_GLOBAL_SCALE_SCREEN'};records=[];work=0
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (time.time()-started>plan['seconds_limit'] or available<544*1024**3
                or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3):
            raise RuntimeError('Global-normalizer CPU screen reserve/cap')
    try:
        guard();torch.set_num_threads(8);torch.set_num_interop_threads(1)
        upstream=a.root/'upstream-parcae-69284c1'
        if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
                or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):
            raise ValueError('Clean pinned independent FP32 reference required')
        sys.path.insert(0,str(upstream));config=ParcaeCandidateConfig(**parent['model'])
        saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        if digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})!=native['full_state_digest']:
            raise AssertionError('Complete mature model/optimizer state readback differs')
        packed=PackedCorpus.from_mapped(a.root/parent['data']['packed'],1024,23,parent['data']['corpus_sha256'])
        if packed.fingerprint!=parent['data']['packing_fingerprint']:raise AssertionError('Frozen natural packing changed')
        if sum(packed.counts(native['source_cursor'],512))!=native['real_targets']:raise AssertionError('Original full-batch normalization differs')
        rows=native['probe_rows'][:8]
        if [r['window'] for r in rows]!=list(range(native['source_cursor'],native['source_cursor']+8)):
            raise AssertionError('Fixed first8 original objective contributions required')
        noise={}
        for rank in {r['rank'] for r in rows}:
            path=capture/f'probe-h0-rank{rank}.npy'
            if file_sha256(path)!=native['noise_sha256_by_rank'][rank]:raise AssertionError('Original supplied latent states changed')
            noise[rank]=np.load(path,mmap_mode='r')
        partial_count=sum(packed.counts(native['source_cursor'],8));gold=None;gold_nll=None;forward_hashes=None
        for mode,scale in [('independent_fp32',1)]+[('cpu_fp16_body_fp32_head',s) for s in plan['loss_scales']]:
            guard();begin=time.time();mixed=mode!='independent_fp32'
            model=ParcaeFP16Candidate(config,initialize=False) if mixed else make_reference(config,True)
            model.load_state_dict(saved['model']);before=digest(dict(sorted(model.state_dict().items())))
            current=None
            if not mixed:model.initialize_state=lambda _:current
            loss=0.;hashes=[]
            with sdpa_kernel(SDPBackend.MATH):
                for row in rows:
                    guard();batch=packed.batch([row['window']],0)[0]
                    current=torch.from_numpy(np.array(noise[row['rank']][row['micro']:row['micro']+1],copy=True))
                    if digest(current)!=row['h0_digest']:raise AssertionError('Exact supplied state differs')
                    n,k=row['n'],row['k']
                    logits=(model(batch.inputs,current,total_steps=n,backward_cap=k) if mixed else
                        model(batch.inputs,return_logits=True,num_steps_pair=torch.tensor([n-k,k]))['logits'])
                    if mixed:hashes.append(digest(logits.detach()))
                    value=(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none')*batch.target_mask.flatten()).sum()
                    (value*(scale/native['real_targets'])).backward();loss+=float(value.detach())
                    work+=matmul_ledger(config,1024,total_steps=n,backward_cap=k)['training_matmul_flops']
            vector={name:p.grad.detach().clone()/scale for name,p in model.named_parameters()}
            finite=all(torch.isfinite(g).all() for g in vector.values())
            if gold is None:
                if not finite:raise FloatingPointError('Nonfinite independent CPU reference')
                gold=vector;gold_nll=loss/partial_count
            comparison=gradient_comparison(gold,vector) if finite else None
            if mixed:
                if forward_hashes is None:forward_hashes=hashes
                elif forward_hashes!=hashes:raise AssertionError('Loss scale changed numerical forward')
            if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('Zero-update screen changed parameters')
            records.append({'precision':mode,'loss_scale':scale,'all_gradients_finite':finite,'gradient':comparison,
                'partial_train_nll':loss/partial_count,'partial_nll_difference_from_fp32':loss/partial_count-gold_nll,
                'partial_real_targets':partial_count,'global_real_token_denominator':native['real_targets'],
                'weights_unchanged':True,'wall_seconds':time.time()-begin})
            write_state(public/'progress.json',{'records':records});del model,vector;gc.collect()
        guard()
        if any(file_sha256(p)!=sha for p,sha in source.items()):raise AssertionError('Frozen scale-screen source changed')
        result={'status':'COMPLETE_CPU_GLOBAL_SCALE_SENSITIVITY','same_fp16_forwards_at_all_scales':True,
            'native_execution_qualified':False,'scientific_training_changed':False}
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(records=records,diagnostic_training_matrix_ops=work,wall_seconds=time.time()-started,
            assigned_chips=0,optimizer_updates=0,test_accessed=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}),flush=True)


if __name__=='__main__':main()
