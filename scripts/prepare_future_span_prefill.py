"""Independent trained-weight references for a conditional,read-only cost check."""
import argparse
import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.future_span_evaluation import prediction_state
from compute_trainium.future_span_prefill import prefill_budget
from compute_trainium.future_span_scientific_state import signature
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_scientific_state import schedule_identity
from compute_trainium.parcae_training_candidate import initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN

IDENTITY='future-span-prefill-preparation-v1'
PUBLIC=Path('results/future_span_prefill_20261025')
ATTEMPT_STARTED=time.time()
RUNTIME={'model_forwards':0,'estimated_forward_matrix_ops':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    def interrupted(signum,frame):raise KeyboardInterrupt('Prefill reference signal '+str(signum))
    for s in [signal.SIGINT,signal.SIGTERM]:signal.signal(s,interrupted)
    root=args.root;out=root/IDENTITY;plan_file=out/'plan.json'
    main_file=root/'future-span-main-v2/plan.json';main=json.loads(main_file.read_text())
    if not plan_file.exists():
        if (root/'future-span-work-control-v1/report.json').exists():
            raise ValueError('Register before work-control outcomes')
        cases={}
        for cell in ['one-ntp','recurrent-ntp']:
            key,case=next((k,c) for k,c in main['cases'].items() if c['cell']==cell and c['seed']==5231)
            file=root/'runs'/f'future-span-main-v2-{key}-finish/checkpoint-u1920.pt'
            native=json.loads((file.parent/'report.json').read_text())
            row=next(r for r in native['checkpoint_records'] if r['update']==1920)
            gate_file=root/'future-span-main-v2/gradients'/key/'report.json'
            gate=json.loads(gate_file.read_text())
            if (gate['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
                    or gate['checkpoint_sha256']!=row['checkpoint_sha256'] or file_sha256(file)!=row['checkpoint_sha256']
                    or native['plan_sha256']!=file_sha256(main_file) or not row['all_complete_replica_states_equal']):
                raise ValueError('Both original trained first-seed models must be complete and qualified')
            cases[key]={'source_case':case,'checkpoint':str(file.relative_to(root)),
                'checkpoint_sha256':row['checkpoint_sha256'],'full_state_digest':row['full_state_digest'],
                'independent_gradient_report_sha256':file_sha256(gate_file),
                'depths':[1] if cell=='one-ntp' else [1,2,3]}
        hardware=root/'future-span-math-qualification-v2/report.json'
        if file_sha256(hardware)!=main['launch_receipts']['future-span-math-qualification-v2']['report_sha256']:
            raise AssertionError('Original warm measurement differs')
        budget=prefill_budget(json.loads(hardware.read_text()))
        if not budget['fits_cap']:raise ValueError('Prospective prefill cost exceeds30min')
        source=dict(main['source_sha256'])
        for f in ['compute_trainium/future_span_prefill.py','tests/test_future_span_prefill.py',
                  'scripts/prepare_future_span_prefill.py','scripts/benchmark_future_span_prefill.py',
                  'scripts/run_future_span_prefill.py','docs/FUTURE_SPAN_PREFILL_20261025.md',
                  'compute_trainium/future_span_work_control.py','compute_trainium/ntp_secondary.py']:
            source[f]=file_sha256(f)
        work_queue=root/'future-span-work-control-queue-v1/plan.json'
        execution=json.loads((root/'future-span-main-v2/execution/execution-start.json').read_text())
        plan=dict(identity=IDENTITY,registered_unix=time.time(),source_sha256=source,
            worker_source_sha256=split_snapshot_sources(source)[0],cases=cases,model=main['model'],
            source_main_plan='future-span-main-v2/plan.json',source_main_plan_sha256=file_sha256(main_file),
            work_queue_plan_sha256=file_sha256(work_queue),budget=budget,
            wait_deadline_unix=execution['deadline_unix']+3*3600,
            cpu_seconds_limit=300,probe_start=1920*512,windows=64,context=1024,
            warm_forwards=100,timed_forwards=100,rounds=2,
            registration_disclosure='Registered after the first seed quality points and first six mature audits,before any matched-work outcome or native head-removed timing. First seed5231 and supported depths are fixed;timing is not a new scientific quality endpoint.',
            scope='Read-only full-logits fixed1024-window prefills,preloaded authentic TRAIN inputs,local batch1 per core. Two32-rank/eight-chip jobs per round;reverse submission order in round2 and retain actual hardware assignment. No future head,backward,optimizer,sampling or KV-cached decoding. Actual weights and original prefix-available latent noise. All cost and failed attempts retained;no TEST or terminal claim.')
        out.mkdir();PUBLIC.mkdir(parents=True)
        write_state(plan_file,plan);write_state(PUBLIC/'preparation-plan.json',plan)
    else:plan=json.loads(plan_file.read_text())
    if args.prepare_only:return
    if (out/'report.json').exists():raise FileExistsError('Preserve completed/stopped prefill reference')
    started=time.time();sha=file_sha256(plan_file);matrix=0
    def guard():
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        if (time.time()-started>plan['cpu_seconds_limit'] or available<544*1024**3
                or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3):
            raise RuntimeError('Read-only CPU prefill reserve/cap')
        if file_sha256(plan_file)!=sha or any(file_sha256(f)!=v for f,v in plan['source_sha256'].items()):
            raise AssertionError('Frozen reference source/plan changed')
    guard();torch.set_num_threads(12);torch.set_num_interop_threads(1)
    upstream=root/'upstream-parcae-69284c1'
    if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
            or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):
        raise ValueError('Clean pinned independent CPU reference required')
    sys.path.insert(0,str(upstream))
    data=main['data'];packed=PackedCorpus.from_mapped(root/data['packed'],1024,23,data['corpus_sha256'])
    if packed.fingerprint!=data['packing_fingerprint']:raise AssertionError('Original source stream differs')
    config=ParcaeCandidateConfig(**plan['model']);seed={x['source_case']['state_seed'] for x in plan['cases'].values()}
    if len(seed)!=1:raise ValueError('Matched original exogenous state seeds required')
    inputs=[];targets=[];masks=[];noise=[]
    for i in range(64):
        window=plan['probe_start']+i;batch=packed.batch([window],0)[0]
        inputs.append(batch.inputs[0]);targets.append(batch.targets[0]);masks.append(batch.target_mask[0])
        noise.append(initial_state(config,window,next(iter(seed)))[0])
    inputs,targets,masks,noise=[torch.stack(v) for v in [inputs,targets,masks,noise]]
    files={}
    for name,value in [('inputs',inputs),('targets',targets),('real-mask',masks),('noise',noise)]:
        f=out/(name+'.npy');np.save(f,value.numpy());files[f.name]=file_sha256(f)
    records=[]
    for key,row in plan['cases'].items():
        guard();file=root/row['checkpoint']
        if file_sha256(file)!=row['checkpoint_sha256']:raise AssertionError('Original trained weights changed')
        saved=torch.load(file,map_location='cpu',mmap=True,weights_only=False);case=row['source_case']
        if (saved['case_id']!=key or saved['update']!=1920 or saved['qualification_only']
                or saved['signature']!=signature(plan['source_main_plan_sha256'],main,case)
                or saved['schedule']!=schedule_identity(case) or saved['stable_trunk']):
            raise ValueError('Exact cooled scientific endpoint required')
        if digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})!=row['full_state_digest']:
            raise AssertionError('Original complete model/optimizer state differs')
        state=prediction_state(saved['model']);expected=digest(dict(sorted(state.items())))
        model=make_reference(config,True).eval().requires_grad_(False);model.load_state_dict(state,strict=True)
        if digest(dict(sorted(model.state_dict().items())))!=expected:raise AssertionError('Head removal changed prediction weights')
        del saved,state;current=None;model.initialize_state=lambda _:current
        for depth in row['depths']:
            samples_file=out/f'{key}-d{depth}-logits.npy'
            samples=np.lib.format.open_memmap(samples_file,mode='w+',dtype=np.float32,shape=(64,8,config.vocab_size))
            sums=[]
            with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                for i in range(64):
                    guard();current=noise[i:i+1]
                    logits=model(inputs[i:i+1],return_logits=True,num_steps_pair=torch.tensor([depth-1,1]))['logits']
                    RUNTIME['model_forwards']+=1
                    RUNTIME['estimated_forward_matrix_ops']+=matmul_ledger(config,1024,total_steps=depth)['forward_matmul_flops']
                    if logits.shape!=(1,1024,config.vocab_size) or not torch.isfinite(logits).all():
                        raise AssertionError('Invalid independent prediction logits')
                    total=0.
                    for first in range(0,1024,128):
                        last=first+128;values=logits[:,first:last].double();target=targets[i:i+1,first:last]
                        total+=float(((values.logsumexp(-1)-values.gather(-1,target[...,None]).squeeze(-1))*masks[i:i+1,first:last]).sum())
                    samples[i]=logits[0,::128].float().numpy();sums.append(total)
                    matrix+=matmul_ledger(config,1024,total_steps=depth)['forward_matmul_flops'];del logits
            samples.flush();del samples
            nll_file=out/f'{key}-d{depth}-nll.npy';np.save(nll_file,np.array(sums,dtype=np.float64))
            for f in [samples_file,nll_file]:files[f.name]=file_sha256(f)
            records.append(dict(case_id=key,depth=depth,logits_file=samples_file.name,nll_file=nll_file.name,
                prediction_state_digest=expected,deployed_parameters=sum(p.numel() for p in model.parameters()),
                head_removed=True,model_unchanged=digest(dict(sorted(model.state_dict().items())))==expected))
        if any(p.grad is not None for p in model.parameters()):raise AssertionError('Read-only CPU reference created gradients')
        del model;gc.collect()
    guard()
    if not all(r['model_unchanged'] and r['deployed_parameters']==config.parameter_count(True) for r in records):
        raise AssertionError('Read-only prediction identity differs')
    if RUNTIME['model_forwards']!=256 or RUNTIME['estimated_forward_matrix_ops']!=matrix:
        raise AssertionError('Complete CPU forward count differs')
    report=dict(status='PASS_INDEPENDENT_HEAD_REMOVED_PREFILL_REFERENCES',plan_sha256=sha,records=records,
        files_sha256=files,windows=64,unique_real_targets=int(masks.sum()),model_forwards=RUNTIME['model_forwards'],
        estimated_forward_matrix_ops=matrix,assigned_chips=0,optimizer_updates=0,test_accessed=False,
        terminal_alpha_spent=0,wall_seconds=time.time()-started,scope=plan['scope'])
    write_state(out/'report.json',report);write_state(PUBLIC/'preparation-report.json',report)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--root',type=Path);args,_=parser.parse_known_args()
        if args.root and (args.root/IDENTITY/'plan.json').exists() and not (args.root/IDENTITY/'report.json').exists():
            result=dict(status='STOPPED_HEAD_REMOVED_PREFILL_REFERENCE',error=str(error),assigned_chips=0,
                optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-ATTEMPT_STARTED,**RUNTIME)
            write_state(args.root/IDENTITY/'report.json',result);write_state(PUBLIC/'preparation-report.json',result)
        raise
