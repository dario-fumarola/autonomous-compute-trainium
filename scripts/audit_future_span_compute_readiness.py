"""Check work/source/cooldown feasibility without opening scientific scores."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from compute_trainium.future_span import FutureSpanCandidate, FutureSpanConfig
from compute_trainium.future_span_control_budget import control_branch
from compute_trainium.future_span_scientific_state import matrix_work
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    source_paths = ['scripts/audit_future_span_compute_readiness.py',
                    'compute_trainium/future_span_control_budget.py',
                    'tests/test_future_span_control_budget.py']
    identity = 'future-span-compute-readiness-v1'
    out = args.root / identity
    public = Path('results/future_span_compute_readiness_20261025')
    if out.exists() or public.exists():
        raise FileExistsError('Preserve an existing readiness audit')
    main_root = args.root / 'future-span-main-v2'
    plan_file = main_root / 'plan.json'
    plan = json.loads(plan_file.read_text())
    if plan['identity'] != 'future-span-main-v2':
        raise ValueError('Exact original MAIN registration required')
    config = ParcaeCandidateConfig(**plan['model'])
    span = FutureSpanConfig(**plan['future_config'])
    # Shape-only model: no weights, data forwards, training RNG or new initializer.
    with torch.device('meta'):
        model = FutureSpanCandidate(config, span, initialize=False, fp16_body=True)
    optimizer = muon_update_matmuls(model.named_parameters(), gate_policy='release_muon')
    packed = args.root / plan['data']['packed']
    manifest_file = packed / 'manifest.json'
    if file_sha256(manifest_file) != plan['data']['manifest_sha256']:
        raise AssertionError('Frozen packing manifest changed')
    manifest = json.loads(manifest_file.read_text())
    prefix_file = packed / 'prefix_counts.npy'
    if file_sha256(prefix_file) != manifest['files']['prefix_counts.npy']:
        raise AssertionError('Original source-count prefix changed')
    counts = np.load(prefix_file, mmap_mode='r')
    tables = {}
    hashes = {}
    for arm in ['one', 'recurrent']:
        case = next(c for c in plan['cases'].values() if c['arm'] == arm)
        for name in ['depth_table', 'work_table']:
            file = main_root / case[name]
            if file_sha256(file) != case[name + '_sha256']:
                raise AssertionError('Frozen execution ledger changed')
            hashes[arm + '-' + name] = case[name + '_sha256']
        depths = np.load(main_root / case['depth_table'], mmap_mode='r')
        work = np.load(main_root / case['work_table'], mmap_mode='r')
        expected = matrix_work(config, span, depths, world=32,
                               optimizer_per_replica=optimizer)
        if not np.array_equal(work, expected):
            raise AssertionError('Work differs from operator/head/replicated-optimizer inventory')
        tables[arm] = work
    candidate = next(c for c in plan['cases'].values() if c['cell'] == 'recurrent-ntp')
    budget = sum(map(int, tables['recurrent'][:candidate['total_updates']]))
    if budget != candidate['training_dense_matrix_ops']:
        raise AssertionError('Candidate complete training budget changed')
    branch = control_branch(tables['one'], counts, budget, origin_update=960)
    result = {'status': 'PASS_PROSPECTIVE_WORK_MATCHED_CONTINUATION_GEOMETRY',
        'identity': identity, 'main_plan_sha256': file_sha256(plan_file),
        'analysis_source_sha256': {p: file_sha256(p) for p in source_paths},
        'frozen_ledger_sha256': hashes, 'source_count_prefix_sha256': file_sha256(prefix_file),
        'model_training_parameters': sum(p.numel() for p in model.parameters()),
        'candidate_updates': candidate['total_updates'],
        'candidate_real_targets': candidate['end_targets'],
        'candidate_training_matrix_ops': budget, 'one_pass_control_branch': branch,
        'eligible_parent_cells': ['one-ntp', 'one-future'],
        'pilot_seeds': [5231, 7283], 'registered_branch': False,
        'checkpoint_adopted': False, 'training_launched': False,
        'assigned_chips': 0, 'model_forwards': 0, 'test_accessed': False,
        'scientific_quality_scores_read': 0, 'terminal_alpha_spent': 0,
        'wall_seconds': time.time() - started,
        'scope': 'Pre-outcome feasibility evidence only, not an admitted experiment. Actual stable parent recovery and cross-plan provenance transport still require qualification, as do stronger ordinary/cheap controls. All source exposure and complete inherited training cost are charged. Existing MAIN source, schedule, budget and analysis remain untouched.'}
    out.mkdir()
    public.mkdir(parents=True)
    write_state(out / 'report.json', result)
    write_state(public / 'report.json', result)
    print(json.dumps({'status': result['status'], 'branch': branch}))


if __name__ == '__main__':
    main()
