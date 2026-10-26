"""Independent actual-weight CPU references for the fixed-curve cost readout."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('USE_TORCH_XLA', '0')
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from compute_trainium.future_span_evaluation import prediction_state
from compute_trainium.future_span_scientific_state import signature
from compute_trainium.parcae_candidate import ParcaeCandidateConfig, matmul_ledger
from compute_trainium.parcae_scientific_state import schedule_identity
from compute_trainium.parcae_training_candidate import initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus, file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    started=time.time();plan=json.loads(a.plan.read_text());out=a.plan.parent
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal CPU reference')
    if plan['purpose']!='fixed_two_prefill_cpu_reference' or plan['optimizer_updates_allowed']!=0:
        raise ValueError('Read-only frozen fixed-curve cost plan required')
    plan_sha=file_sha256(a.plan);torch.set_num_threads(12);torch.set_num_interop_threads(1)
    def guard():
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        if (time.time()-started>300 or available<544*1024**3
                or min(shutil.disk_usage('/').free,shutil.disk_usage(out).free)<35*1024**3):
            raise RuntimeError('CPU prefill reference cap or resource reserve')
        if file_sha256(a.plan)!=plan_sha or any(file_sha256(f)!=h for f,h in plan['source_sha256'].items()):
            raise AssertionError('Frozen read-only reference changed')
    guard();upstream=a.root/'upstream-parcae-69284c1'
    if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
            or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):
        raise AssertionError('Clean pinned independent upstream required')
    sys.path.insert(0,str(upstream))
    original_file=a.root/plan['source_main_plan'];original=json.loads(original_file.read_text())
    if file_sha256(original_file)!=plan['source_main_plan_sha256']:
        raise AssertionError('Original fixed-curve plan changed')
    data=original['data'];packed=PackedCorpus.from_mapped(a.root/data['packed'],1024,23,data['corpus_sha256'])
    if packed.fingerprint!=data['packing_fingerprint']:raise AssertionError('Original TRAIN stream changed')
    config=ParcaeCandidateConfig(**plan['model']);seeds={v['source_case']['state_seed'] for v in plan['cases'].values()}
    if len(seeds)!=1:raise ValueError('Matched original prefix-available state noise required')
    arrays={k:[] for k in ['inputs','targets','real-mask','noise']}
    for i in range(64):
        window=plan['probe_start']+i;batch=packed.batch([window],0)[0]
        for name,value in [('inputs',batch.inputs[0]),('targets',batch.targets[0]),('real-mask',batch.target_mask[0]),
                           ('noise',initial_state(config,window,next(iter(seeds)))[0])]:
            arrays[name].append(value)
    arrays={k:torch.stack(v) for k,v in arrays.items()};files={}
    for name,value in arrays.items():
        f=out/(name+'.npy');np.save(f,value.numpy());files[f.name]=file_sha256(f)
    records=[];matrix=0
    for key,row in plan['cases'].items():
        guard();checkpoint=a.root/row['checkpoint'];case=row['source_case']
        if file_sha256(checkpoint)!=row['checkpoint_sha256']:raise AssertionError('Actual trained weights changed')
        saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        if (saved['case_id']!=key or saved['update']!=case['total_updates'] or saved['qualification_only']
                or saved['stable_trunk'] or saved['signature']!=signature(plan['source_main_plan_sha256'],original,case)
                or saved['schedule']!=schedule_identity(case)):
            raise ValueError('Exact completed cooldown and scientific history required')
        if digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})!=row['full_state_digest']:
            raise AssertionError('Complete trained state differs')
        state=prediction_state(saved['model']);before=digest(dict(sorted(state.items())))
        model=make_reference(config,True).eval().requires_grad_(False);model.load_state_dict(state,strict=True)
        del saved,state;current=None;model.initialize_state=lambda _:current
        for depth in row['depths']:
            f=out/f'{key}-d{depth}-logits.npy'
            samples=np.lib.format.open_memmap(f,mode='w+',dtype=np.float32,shape=(64,8,config.vocab_size));sums=[]
            with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                for i in range(64):
                    guard();current=arrays['noise'][i:i+1]
                    logits=model(arrays['inputs'][i:i+1],return_logits=True,num_steps_pair=torch.tensor([depth-1,1]))['logits']
                    if logits.shape!=(1,1024,config.vocab_size) or not torch.isfinite(logits).all():
                        raise AssertionError('Independent prediction shape/finiteness failed')
                    total=0.
                    for start in range(0,1024,128):
                        stop=start+128;values=logits[:,start:stop].double();targets=arrays['targets'][i:i+1,start:stop]
                        total+=float(((values.logsumexp(-1)-values.gather(-1,targets[...,None]).squeeze(-1))*arrays['real-mask'][i:i+1,start:stop]).sum())
                    samples[i]=logits[0,::128].float().numpy();sums.append(total)
                    matrix+=matmul_ledger(config,1024,total_steps=depth)['forward_matmul_flops'];del logits
            samples.flush();del samples
            nll=out/f'{key}-d{depth}-nll.npy';np.save(nll,np.asarray(sums,dtype=np.float64))
            for path in [f,nll]:files[path.name]=file_sha256(path)
            records.append(dict(case_id=key,depth=depth,logits_file=f.name,nll_file=nll.name,
                prediction_state_digest=before,deployed_parameters=sum(v.numel() for v in model.parameters()),
                head_removed=True,model_unchanged=digest(dict(sorted(model.state_dict().items())))==before))
        if any(v.grad is not None for v in model.parameters()):raise AssertionError('Read-only reference created gradients')
        del model;gc.collect()
    guard()
    if not all(r['model_unchanged'] and r['deployed_parameters']==config.parameter_count(True) for r in records):
        raise AssertionError('Prediction state/count changed')
    write_state(out/'report.json',dict(status='PASS_INDEPENDENT_HEAD_REMOVED_PREFILL_REFERENCES',
        plan_sha256=plan_sha,records=records,files_sha256=files,wall_seconds=time.time()-started,
        model_forwards=128,estimated_forward_matrix_ops=matrix,optimizer_updates=0,assigned_chips=0,
        test_accessed=False,scope=plan['scope']))


if __name__=='__main__':main()
