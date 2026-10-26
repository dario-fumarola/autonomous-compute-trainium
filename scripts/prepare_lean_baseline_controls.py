"""Random initializers and counted endpoints only; no predictive fit admission."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from compute_trainium.future_span_evaluation import prediction_state
from compute_trainium.lean_mixing_precision import LeanFP32MixingCandidate
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_mixing_control import mixing_only_ledger
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
from compute_trainium.parcae_scientific_schedule import matched_endpoint,source_endpoint
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint,file_sha256
from scripts.check_ordinary8_cpu import ordinary_reference_config
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN,PlainTiedHead
from scripts.run_fixed_control_replication import reserves

IDENTITY='lean-baseline-control-preparation-v2'
PUBLIC=Path('results/lean_baseline_controls_20261026/preparation-v2')
DOC='docs/LEAN_BASELINE_PREDICTIVE_PLAN_20261026.md'


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/IDENTITY
    if out.exists():raise FileExistsError('Preserve original initializer preparation')
    reserves(a.root)
    if shutil.disk_usage(a.root).free<42*1024**3:raise RuntimeError('Initializer/table preparation plus protected space required')
    torch.set_num_threads(12);torch.set_num_interop_threads(1)
    qualification_file=a.root/'baseline-numerical-recovery-v1/report.json';qualified=read(qualification_file)
    if qualified['status']!='PASS_THREE_BASELINE_NUMERICAL_RECOVERY_PROFILES':raise ValueError('All three baseline numerical profiles required')
    original_file=a.root/'fixed-two-ntp-main-v1/plan.json';original=read(original_file)
    paired_file=a.root/'fixed-control-replication-main-v1/full-plan.json';paired=read(paired_file)
    tune_file=a.root/'fixed-two-mixing-tune-v1/plan.json';tune=read(tune_file)
    model=ParcaeCandidateConfig(**original['model']);ordinary=ParcaeCandidateConfig(**read(a.root/'ordinary8-cpu-parity-v2/report.json')['model'])
    if paired['model']!=original['model'] or tune['model']!=original['model']:raise AssertionError('Paired/tuning random family differs')
    data=original['data'];packing=a.root/data['packed'];meta=read(packing/'manifest.json')
    if file_sha256(packing/'manifest.json')!=data['manifest_sha256'] or file_sha256(packing/'prefix_counts.npy')!=meta['files']['prefix_counts.npy']:
        raise AssertionError('Accepted natural source/counts changed')
    counts=np.load(packing/'prefix_counts.npy',mmap_mode='r');source_total=int((counts[-1]-counts[0]).sum())
    if source_total!=6500009163:raise AssertionError('Original accepted6.5B source extent required')
    budget=original['cases']['fixed-2b']['training_dense_matrix_ops']
    if budget!=1272689958300549120:raise AssertionError('Original fixed-two2B work target changed')
    profiles={}
    for role in ['one','mixing','ordinary8']:
        c=ordinary if role=='ordinary8' else model
        with torch.device('meta'):
            target=(LeanFP32MixingCandidate(c,initialize=False) if role=='mixing' else
                    ParcaePrimitiveSDPACandidate(c,recurrent=role!='ordinary8',initialize=False))
        optimizer=32*muon_update_matmuls(target.named_parameters(),gate_policy='release_muon')
        neural=(mixing_only_ledger(c,1024,512,total_steps=2,backward_cap=2) if role=='mixing' else
                matmul_ledger(c,1024,512,recurrent=role!='ordinary8',total_steps=1,backward_cap=1))['training_matmul_flops']
        total=neural+optimizer;native=qualified['records'][role]['warm']
        if native['model_matmuls']!=neural*121 or native['optimizer_matmuls']!=optimizer*121:
            raise AssertionError('Static accounting disagrees with actual numerical trainer counters: '+role)
        parameters=sum(p.numel() for p in target.parameters());updates=budget//total
        rows=updates+2;depth=2 if role=='mixing' else 1
        depths=np.full((rows,16,32,2),depth,dtype=np.uint16);work=np.full(rows,total,dtype=np.int64)
        endpoint=matched_endpoint(work,budget);endpoint.update(source_endpoint(counts,updates,512),
            cooldown_start_update=updates//2,cooldown_start_targets=source_endpoint(counts,updates//2,512)['real_targets'])
        profiles[role]=dict(model=dict(c.__dict__),parameters=parameters,depth=depth,
            arm='ordinary' if role=='ordinary8' else ('poisson2' if role=='mixing' else 'fixed1'),
            model_forward_backward_matmuls=neural,replicated_optimizer_matmuls=optimizer,total_update_work=total,
            matched_work=endpoint,tuning_updates=576,tuning_targets=source_endpoint(counts,576,512)['real_targets'],
            tuning_cooldown_start_targets=source_endpoint(counts,288,512)['real_targets'])
        del target,depths,work
    upstream=a.root/'upstream-parcae-69284c1'
    if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
            or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):raise ValueError('Clean pinned random initializer required')
    sys.path.insert(0,str(upstream));import parcae_lm.models.gpt.gpt as gpt
    gpt.LinearCrossEntropyLoss=PlainTiedHead
    sources={**read(a.root/'baseline-numerical-recovery-v1/plan.json')['source_sha256']}
    for name in ['scripts/prepare_lean_baseline_controls.py',DOC]:sources[name]=file_sha256(name)
    if any(file_sha256(name)!=sha for name,sha in sources.items()):raise AssertionError('Qualified preparation sources changed')
    out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=True);initials={};status=dict(status='STOPPED_LEAN_BASELINE_INITIALIZER_PREPARATION');written=[]
    try:
        for seed in [4127,5231,17293]:
            reserves(a.root)
            if time.time()-started>900:raise TimeoutError('Bounded initializer CPU preparation')
            family_plan=tune if seed==4127 else paired;parent=tune_file.parent if seed==4127 else paired_file.parent
            initial=family_plan['initial_states'][f's{seed}-family'];source=parent/initial['file']
            if file_sha256(source)!=initial['sha256']:raise AssertionError('Original paired random initializer changed')
            saved=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
            if saved['qualification_only'] is not False or saved.get('seed',saved.get('backbone_model_seed'))!=seed:raise ValueError('Only original random scientific initialization may be cloned')
            state=prediction_state(saved['model']);key=f's{seed}-family';file=out/(key+'.pt');model_digest=digest(dict(sorted(state.items())))
            atomic_checkpoint(file,dict(model=state,config=original['model'],qualification_only=False,seed=seed,
                model_digest=model_digest,purpose='fresh_random_lean_baseline_initializer',optimizer_updates=0,
                original_full_initializer_sha256=initial['sha256']))
            check=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
            if digest(dict(sorted(check['model'].items())))!=model_digest:raise AssertionError('Unchanged paired body disk readback failed')
            initials[key]=dict(file=file.name,sha256=file_sha256(file),model_digest=model_digest,parameters=141963840,
                seed=seed,source_initializer_sha256=initial['sha256']);written.append(key)
            del saved,state,check;gc.collect()
            torch.manual_seed(seed);fresh=ordinary_reference_config(ordinary).construct_model().train()
            if sum(p.numel() for p in fresh.parameters())!=141733120:raise AssertionError('Eight-layer random inventory differs')
            key=f's{seed}-ordinary8';file=out/(key+'.pt');state=fresh.state_dict();model_digest=digest(dict(sorted(state.items())))
            atomic_checkpoint(file,dict(model=state,config=dict(ordinary.__dict__),qualification_only=False,seed=seed,
                model_digest=model_digest,purpose='fresh_random_ordinary8_baseline_initializer',optimizer_updates=0,upstream_commit=PIN))
            check=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
            if digest(dict(sorted(check['model'].items())))!=model_digest:raise AssertionError('Fresh ordinary8 disk readback failed')
            initials[key]=dict(file=file.name,sha256=file_sha256(file),model_digest=model_digest,parameters=141733120,seed=seed)
            written.append(key);del fresh,state,check;gc.collect()
        for role,profile in profiles.items():
            rows=profile['matched_work']['updates']+2;depth=np.full((rows,16,32,2),profile['depth'],dtype=np.uint16)
            work=np.full(rows,profile['total_update_work'],dtype=np.int64)
            for label,value in [('depths',depth),('work',work)]:
                file=out/(role+'-'+label+'.npy');np.save(file,value);profile[label+'_file']=file.name;profile[label+'_sha256']=file_sha256(file)
        manifest=dict(status='PASS_LEAN_BASELINE_CONTROL_CPU_PREPARATION_ONLY',identity=IDENTITY,
            source_sha256=sources,profiles=profiles,initial_states=initials,data=data,evaluation=original['evaluation'],
            source_total_real_targets=source_total,target_training_matrix_budget=budget,seeds=[5231,17293],tuning_seed=4127,
            lr_grid=[.008,.016,.032],qualification_report_sha256=file_sha256(qualification_file),
            original_fixed_plan_sha256=file_sha256(original_file),model_forwards=0,optimizer_updates=0,
            assigned_chips=0,scientific_fit_admitted=False,test_accessed=False,terminal_alpha_spent=0,
            wall_seconds=time.time()-started,scope=Path(DOC).read_text())
        write_state(out/'manifest.json',manifest);write_state(PUBLIC/'manifest.json',manifest);status=manifest
    except BaseException as error:status.update(error=str(error),error_kind=type(error).__name__,completed_initializer_keys=written,
        model_forwards=0,optimizer_updates=0,assigned_chips=0,wall_seconds=time.time()-started,source_sha256=sources);raise
    finally:write_state(out/'report.json',status);write_state(PUBLIC/'report.json',status)


if __name__=='__main__':main()
