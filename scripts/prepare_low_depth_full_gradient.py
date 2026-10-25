"""Prepare the one reserved gradient-rule comparison; never launches it."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch

from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig
from compute_trainium.parcae_scientific_schedule import update_work,matched_endpoint,source_endpoint,cooldown_start
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/'parcae-full-gradient-preparation-v1';public=Path('results/parcae_full_gradient_preparation_20261025')
    if out.exists():raise FileExistsError('No silent full-gradient preparation replacement')
    parent_dir=a.root/'parcae-low-depth-pilot-v1';parent_file=parent_dir/'preparation-plan.json'
    parent=json.loads(parent_file.read_text());candidate=next(r for r in parent['records'] if r['arm']=='poisson2')
    for path,sha in [(parent_dir/candidate['depth_table'],candidate['depth_table_sha256']),
                     (parent_dir/candidate['work_table'],candidate['work_table_sha256'])]:
        if file_sha256(path)!=sha:raise AssertionError('Original sampled forward/work tables changed')
    packed=a.root/'natural-append-6500m-v2/packed';manifest=packed/'manifest.json'
    if file_sha256(manifest)!=parent['packed_manifest_sha256']:raise AssertionError('Frozen natural packing manifest changed')
    metadata=json.loads(manifest.read_text());prefix_file=packed/'prefix_counts.npy'
    if file_sha256(prefix_file)!=metadata['files']['prefix_counts.npy']:raise AssertionError('Frozen real-source count map changed')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<544*1024**3 or min(shutil.disk_usage(a.root).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Preparation memory/disk reserve')
    original=np.load(parent_dir/candidate['depth_table'],mmap_mode='r');full=np.array(original,copy=True)
    full[...,1]=full[...,0]
    if not np.array_equal(original[...,0],full[...,0]):raise AssertionError('Gradient-only preparation changed forward counts')
    c=ParcaeCandidateConfig(**parent['model'])
    with torch.device('meta'):model=ParcaeCandidate(c,recurrent=True,initialize=False)
    optimizer=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')
    work=update_work(c,full,recurrent=True,context=1024,world=32,optimizer_per_replica=optimizer)
    prefix=np.load(prefix_file,mmap_mode='r');endpoint=matched_endpoint(work,parent['reference_budget_matmuls'])
    endpoint.update(source_endpoint(prefix,endpoint['updates'],512));mid=cooldown_start(endpoint['updates'])
    endpoint.update(cooldown_start_update=mid,cooldown_start_targets=source_endpoint(prefix,mid,512)['real_targets'])
    equal={**source_endpoint(prefix,1920,512),'updates':1920,'cooldown_start_update':960,
        'cooldown_start_targets':source_endpoint(prefix,960,512)['real_targets'],'executed_matmuls':int(work[:1920].sum())}
    if mid<=1920:raise AssertionError('Stable1B point must precede matched-work cooldown')
    out.mkdir();public.mkdir(parents=True,exist_ok=False);np.save(out/'full-depths.npy',full);np.save(out/'full-work.npy',work)
    plan={'identity':'parcae-full-gradient-preparation-v1','registered_unix':started,
        'parent_preparation_sha256':file_sha256(parent_file),'model':parent['model'],
        'same_forward_table_sha256':candidate['depth_table_sha256'],'full_table_sha256':file_sha256(out/'full-depths.npy'),
        'full_work_sha256':file_sha256(out/'full-work.npy'),'forward_counts_exactly_equal':True,
        'backward_rule':'Differentiate every sampled forward application;no resampling of forward depth.',
        'initial_state':{**parent['initial_states']['s1231-family'],'file':'../parcae-low-depth-pilot-v1/'+parent['initial_states']['s1231-family']['file']},
        'seed':1231,'matched_work':endpoint,'equal_data':equal,'ordinary_reference_budget':parent['reference_budget_matmuls'],
        'truncated_reference':candidate['matched_work'],'parameters':candidate['parameters'],
        'learning_rate_policy':'Exactly the scalar rate selected for the truncated candidate;no new tuning in the gradient-only causal screen.',
        'source_sha256':{f:file_sha256(f) for f in ['scripts/prepare_low_depth_full_gradient.py',
            'compute_trainium/parcae_scientific_schedule.py','compute_trainium/parcae_work_accounting.py',
            'compute_trainium/parcae_candidate.py']},
        'scientific_launch_authorized':False,'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,
        'scope':'CPU numerical depth/work preparation for the single reserved conditional follow-up. Same N,initial weights,source/noise order,architecture and readout;only K changes from1 toN. No new scientific fit,heldout score or automatic advancement. Own numerical/profile gates and conditional scientific registration remain required.'}
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    result={'status':'COMPLETE_UNLAUNCHED_FULL_GRADIENT_PREPARATION','matched_work':endpoint,'equal_data':equal,
        'forward_counts_exactly_equal':True,'scientific_launch_authorized':False,'assigned_chips':0,'optimizer_updates':0,
        'wall_seconds':time.time()-started,'test_accessed':False,'scope':plan['scope']}
    write_state(out/'report.json',result);write_state(public/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
