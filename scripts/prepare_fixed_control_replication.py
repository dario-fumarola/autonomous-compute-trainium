"""CPU-only initializer/ledger preparation for a stronger fixed-depth pilot."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch
from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig
from compute_trainium.mixing_control_science import matrix_work as mixing_work
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint,file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN

IDENTITY='fixed-control-replication-preparation-v1'
PUBLIC=Path('results/fixed_control_replication_20261026/preparation')
DOC='docs/FIXED_CONTROL_REPLICATION_PREPARATION_20261026.md'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    started=time.time();out=a.root/IDENTITY;out.mkdir();PUBLIC.mkdir(parents=True)
    torch.set_num_threads(12);torch.set_num_interop_threads(1)
    available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available<544*1024**3 or min(shutil.disk_usage('/').free,shutil.disk_usage(out).free)<35*1024**3:
        raise RuntimeError('CPU preparation resource reserve')
    parent_file=a.root/'fixed-two-ntp-main-v1/plan.json';parent=json.loads(parent_file.read_text())
    complete=a.root/'fixed-two-ntp-main-v1/report.json'
    if json.loads(complete.read_text())['status']!='COMPLETE_FIXED_TWO_NTP_CURVE':raise ValueError('Complete existing curve required')
    sources={**parent['source_sha256'],**{f:file_sha256(f) for f in [__file__,DOC]}}
    # Executable manifests use repository-relative paths,including this entry.
    sources.pop(__file__,None);sources['scripts/prepare_fixed_control_replication.py']=file_sha256(__file__)
    if any(file_sha256(f)!=h for f,h in sources.items()):raise AssertionError('Original qualified source changed')
    fresh_seed=17293;previous=set()
    def inspect(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ['seed','model_seed','backbone_model_seed'] and type(item) is int:previous.add(item)
                inspect(item)
        elif isinstance(value,list):
            for item in value:inspect(item)
    for file in Path('results').rglob('*.json'):
        inspect(json.loads(file.read_text()))
    if fresh_seed in previous:raise ValueError('Chosen development initializer seed was already used')
    upstream=a.root/'upstream-parcae-69284c1'
    if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
            or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):
        raise ValueError('Clean pinned upstream random initializer required')
    sys.path.insert(0,str(upstream));config=ParcaeCandidateConfig(**parent['model']);span=FutureSpanConfig(**parent['future_config'])
    torch.manual_seed(fresh_seed);reference=make_reference(config,True)
    body_digest=digest(dict(sorted(reference.state_dict().items())))
    model=FutureSpanCandidate(config,span,head_seed=104729+fresh_seed,initialize=False,fp16_body=True)
    missing=model.load_state_dict(reference.state_dict(),strict=False)
    if set(missing.missing_keys)!={'future_lm_head.weight','future_lm_head.bias','future_positions'} or missing.unexpected_keys:
        raise AssertionError('New original initializer changes the prediction inventory')
    if digest(dict(sorted(model.prediction_state_dict().items())))!=body_digest:raise AssertionError('Fresh prediction weights changed')
    model_digest=digest(dict(sorted(model.state_dict().items())));initial=out/f'initial-s{fresh_seed}.pt'
    atomic_checkpoint(initial,dict(model=model.state_dict(),seed=fresh_seed,head_seed=104729+fresh_seed,
        config=asdict(config),future_config=asdict(span),qualification_only=False,
        purpose='fresh_random_future_span_scientific_initializer',model_digest=model_digest,
        upstream_commit=PIN,optimizer_updates=0))
    check=torch.load(initial,map_location='cpu',mmap=True,weights_only=False)
    if check['optimizer_updates']!=0 or digest(dict(sorted(check['model'].items())))!=model_digest:
        raise AssertionError('Original initializer readback differs')
    optimizer_ops=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')
    initial_info=dict(file=str(initial.relative_to(a.root)),sha256=file_sha256(initial),model_digest=model_digest,
        backbone_digest=body_digest,seed=fresh_seed,head_seed=104729+fresh_seed,
        training_parameters=sum(v.numel() for v in model.parameters()),deployed_parameters=config.parameter_count(True))
    del reference,model,check;gc.collect()
    original=parent['initial_states']['s5231-family'];original_file=parent_file.parent/original['file']
    if file_sha256(original_file)!=original['sha256']:raise AssertionError('Existing original initializer changed')
    initial_states={'s5231-family':{**original,'file':str(original_file.resolve().relative_to(a.root))},
                    f's{fresh_seed}-family':initial_info}
    table=parent_file.parent/parent['cases']['fixed-2b']['depth_table']
    if file_sha256(table)!=parent['cases']['fixed-2b']['depth_table_sha256']:raise AssertionError('Fixed depth geometry changed')
    depths=np.load(table,mmap_mode='r');work=mixing_work(config,span,depths,world=32,optimizer_per_replica=optimizer_ops)
    if not np.all(depths==2) or not np.all(work==work[0]):raise AssertionError('Uniform mixing geometry required')
    np.save(out/'mixing-work.npy',work);mixing_per_update=int(work[0]);target_work=parent['cases']['fixed-2b']['training_dense_matrix_ops']
    mix_updates=target_work//mixing_per_update
    if not mixing_per_update*mix_updates<=target_work<mixing_per_update*(mix_updates+1):raise AssertionError('Exact work floor required')
    packing=a.root/parent['data']['packed'];metadata=json.loads((packing/'manifest.json').read_text())
    if file_sha256(packing/'manifest.json')!=parent['data']['manifest_sha256']:raise AssertionError('Original source packing changed')
    prefix_file=packing/'prefix_counts.npy'
    if file_sha256(prefix_file)!=metadata['files']['prefix_counts.npy']:raise AssertionError('Original source counters changed')
    counts=np.load(prefix_file,mmap_mode='r');cases=[]
    for name,seed,variant,updates in [('cheap-s5231-work',5231,'mixing',mix_updates),
        (f'fixed-s{fresh_seed}-data',fresh_seed,'fixed',3840),
        (f'one-s{fresh_seed}-data',fresh_seed,'one',3840),
        (f'one-s{fresh_seed}-work',fresh_seed,'one',4625),
        (f'cheap-s{fresh_seed}-work',fresh_seed,'mixing',mix_updates)]:
        cases.append(dict(case_id=name,seed=seed,variant=variant,total_updates=int(updates),
            real_targets=int((counts[updates*512]-counts[0]).sum()),
            cooldown_start_update=int(updates//2),cooldown_start_targets=int((counts[(updates//2)*512]-counts[0]).sum()),
            peak_lr=None if variant=='mixing' else .032,
            rate_scope='Select only from the separately frozen fixed-mixing TUNE' if variant=='mixing' else 'Original independent three-rate selection;no per-seed retuning'))
    if time.time()-started>180:raise TimeoutError('CPU preparation cap')
    result=dict(status='PASS_FIXED_CONTROL_REPLICATION_CPU_PREPARATION_ONLY',identity=IDENTITY,
        source_sha256=sources,parent_plan_sha256=file_sha256(parent_file),parent_report_sha256=file_sha256(complete),
        model=parent['model'],future_config=parent['future_config'],data=parent['data'],evaluation=parent['evaluation'],
        initial_states=initial_states,fresh_development_seed=fresh_seed,previous_model_seeds_checked=len(previous),
        fresh_confirmation=False,cases=cases,reference_target_matrix_work=target_work,
        mixing_matrix_work_per_update=mixing_per_update,mixing_work_sha256=file_sha256(out/'mixing-work.npy'),
        proposed_actual_updates=int(2*mix_updates+3840+4625+1920),
        shared_prefix_scope='New One data/work endpoints may share the stable1920-update prefix;all initializers are untrained.',
        optimizer_updates=0,model_forwards=0,assigned_chips=0,native_launch_allowed=False,
        scientific_main_admitted=False,test_accessed=False,terminal_alpha_spent=0,wall_seconds=time.time()-started,
        scope=Path(DOC).read_text())
    write_state(out/'manifest.json',result);write_state(PUBLIC/'report.json',result)
    print(json.dumps(dict(status=result['status'],fresh_seed=fresh_seed,mixing_updates=int(mix_updates),
        actual_updates=result['proposed_actual_updates'],wall_seconds=result['wall_seconds'])))


if __name__=='__main__':
    process_start=time.time()
    try:main()
    except BaseException as error:
        p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path);a,_=p.parse_known_args()
        result=dict(status='STOPPED_FIXED_CONTROL_REPLICATION_PREPARATION',error=str(error),
            optimizer_updates=0,model_forwards=0,assigned_chips=0,native_launch_allowed=False,
            test_accessed=False,wall_seconds=time.time()-process_start)
        if a.root and (a.root/IDENTITY).exists() and not (a.root/IDENTITY/'manifest.json').exists():
            write_state(a.root/IDENTITY/'failure.json',result)
        PUBLIC.mkdir(parents=True,exist_ok=True);write_state(PUBLIC/'failure.json',result)
        raise
