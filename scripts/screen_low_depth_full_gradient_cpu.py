"""TRAIN-only precision feasibility for full gradients at a trained checkpoint."""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA', '0')
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from compute_trainium.depth_mixture import routed_indices
from compute_trainium.parcae_candidate import ParcaeCandidateConfig, matmul_ledger
from compute_trainium.parcae_fp16_candidate import ParcaeFP16Candidate
from compute_trainium.parcae_scientific_state import signature, validate_checkpoint
from compute_trainium.parcae_training_candidate import gradient_comparison, initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus, file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    experiment = args.root / 'parcae-low-depth-pilot-v1'
    parent_file = experiment / 'main-plan.json'
    parent_sha = file_sha256(parent_file)
    parent = json.loads(parent_file.read_text())
    case_id = 'poisson2-main-s1231'
    case = parent['cases'][case_id]
    checkpoint = args.root / 'runs/low-depth-main-v1-poisson2-first/checkpoint-u576.pt'
    event_file = args.root / 'logs/low-depth-main-v1-poisson2-first.log'
    events = [json.loads(s) for s in event_file.read_text().splitlines() if s.startswith('{')]
    event = next(r for r in events if r.get('event') == 'checkpoint' and r['update'] == 576)
    if not event['all_complete_replica_states_equal']:
        raise ValueError('Completed all-replica checkpoint proof required')
    out = args.root / 'low-depth-full-gradient-cpu-screen-v1'
    public = Path('results/low_depth_full_gradient_cpu_20261025')
    out.mkdir(exist_ok=False); public.mkdir(parents=True, exist_ok=False)
    sources = {**parent['source_sha256'], __file__: file_sha256(__file__)}
    # Keep the public source manifest relative to the repository.
    sources.pop(__file__)
    sources['scripts/screen_low_depth_full_gradient_cpu.py'] = file_sha256(__file__)
    cursor = 576*512
    depths_file = experiment / case['depth_table']
    if file_sha256(depths_file) != case['depth_table_sha256']:
        raise AssertionError('Main sampled forward counts changed')
    depths = np.load(depths_file, mmap_mode='r')
    layout = {}
    for micro in range(16):
        for rank in range(32):
            window = routed_indices(cursor, micro, rank, 32, 1, 16, parent['routing_seed'])[0]
            if window < cursor+8:
                n, k = map(int, depths[576, micro, rank])
                if k != 1:
                    raise AssertionError('Original candidate must use last-one truncation')
                layout[window] = {'window': window, 'forward_steps': n}
    if set(layout) != set(range(cursor, cursor+8)):
        raise AssertionError('First eight original next-TRAIN windows must be present exactly once')
    plan = {'identity': 'low-depth-full-gradient-cpu-screen-v1', 'registered_unix': started,
            'main_plan_sha256': parent_sha, 'checkpoint_sha256': event['checkpoint_sha256'],
            'training_update': 576, 'source_cursor': cursor, 'windows': list(layout[w] for w in sorted(layout)),
            'source_sha256': sources, 'upstream_commit': PIN, 'precision_cases': ['independent_cpu_fp32', 'cpu_fp16_body_fp32_head_scaled8192'],
            'gradient_rules': ['last_one', 'all_sampled_steps'], 'gradient_relative_gate': .02,
            'loss_absolute_gate': .003, 'seconds_limit': 1200, 'assigned_chips': 0,
            'optimizer_updates': 0, 'test_accessed': False,
            'scope': 'Numerical feasibility at an immutable301M trained checkpoint on the first8 '
            'next natural TRAIN windows,selected without scoring. Same weights,forward depths,source '
            'order,initial states and loss;only differentiated tail and numerical precision vary. '
            'No parameter updates or held-out data. Small CPU subbatch,not a complete512-window '
            'native qualification,learning result or permission to launch the conditional follow-up.'}
    write_state(out / 'plan.json', plan); write_state(public / 'plan.json', plan)
    records = []; vectors = {}; hashes = {}; result = {'status': 'STOPPED_FULL_GRADIENT_CPU_SCREEN'}

    def guard():
        if time.time()-started > plan['seconds_limit']:
            raise TimeoutError('Original full-gradient CPU screen cap')
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))
        if available < 544*1024**3 or min(shutil.disk_usage(out).free, shutil.disk_usage('/').free) < 35*1024**3:
            raise RuntimeError('Full-gradient CPU screen resource reserve')

    try:
        guard(); torch.set_num_threads(12); torch.set_num_interop_threads(1)
        upstream = args.root / 'upstream-parcae-69284c1'
        if (subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != PIN
                or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()):
            raise ValueError('Clean pinned upstream FP32 reference required')
        sys.path.insert(0, str(upstream))
        if file_sha256(checkpoint) != plan['checkpoint_sha256']:
            raise AssertionError('Trained checkpoint changed')
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        full_digest = digest({'model': dict(sorted(saved['model'].items())), 'optimizer': saved['optimizer']})
        if full_digest != saved['model_optimizer_digest'] or full_digest != event['full_state_digest']:
            raise AssertionError('Independent complete-state readback differs')
        packed = PackedCorpus.from_mapped(args.root / parent['data']['packed'], 1024, 23,
                                          parent['data']['corpus_sha256'])
        work_file = experiment / case['work_table']
        if file_sha256(work_file) != case['work_table_sha256']:
            raise AssertionError('Main work table changed')
        validate_checkpoint(saved, case_id, case, signature(parent_sha, parent, case),
                            expected_targets=sum(packed.counts(0, cursor)),
                            expected_matmuls=int(np.load(work_file, mmap_mode='r')[:576].sum()))
        config = ParcaeCandidateConfig(**parent['model'])
        total = sum(packed.counts(cursor, 8))
        diagnostic_work = 0
        for rule in ['last_one', 'all_sampled_steps']:
            for mixed in [False, True]:
                guard(); begin = time.time(); label = 'fp16' if mixed else 'fp32'
                model = (ParcaeFP16Candidate(config, recurrent=True, initialize=False) if mixed
                         else make_reference(config, True))
                model.load_state_dict(saved['model'], strict=True)
                before = digest(dict(sorted(model.state_dict().items())))
                current = None
                if not mixed:
                    model.initialize_state = lambda _: current.clone()
                loss = 0.; scale = 8192 if mixed else 1; output_hashes = []
                with sdpa_kernel(SDPBackend.MATH):
                    for window, row in sorted(layout.items()):
                        guard(); n = row['forward_steps']; k = 1 if rule == 'last_one' else n
                        batch = packed.batch([window], 0)[0]
                        current = initial_state(config, window, case['state_seed'])
                        logits = (model(batch.inputs, current.clone(), total_steps=n, backward_cap=k) if mixed
                                  else model(batch.inputs, return_logits=True,
                                             num_steps_pair=torch.tensor([n-k, k]))['logits'])
                        output_hashes.append(digest(logits.detach()))
                        value = (F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), reduction='none')
                                 * batch.target_mask.flatten()).sum()
                        (value*(scale/total)).backward(); loss += float(value.detach())
                        diagnostic_work += matmul_ledger(config, 1024, 1, total_steps=n,
                                                         backward_cap=k)['training_matmul_flops']
                gradient = {name: p.grad.detach().clone()/scale for name, p in model.named_parameters()}
                if any(not torch.isfinite(g).all() for g in gradient.values()):
                    raise ValueError('Nonfinite CPU full-gradient precision probe')
                if digest(dict(sorted(model.state_dict().items()))) != before:
                    raise AssertionError('Zero-update probe changed model weights')
                vectors[rule, label] = gradient; hashes[rule, label] = output_hashes
                records.append({'rule': rule, 'precision': label, 'nll': loss/total,
                                'weights_unchanged': True, 'wall_seconds': time.time()-begin})
                write_state(public / 'progress.json', {'records': records})
                del model; gc.collect()
        comparisons = {}
        for rule in ['last_one', 'all_sampled_steps']:
            comparison = gradient_comparison(vectors[rule, 'fp32'], vectors[rule, 'fp16'])
            loss_difference = next(r['nll'] for r in records if (r['rule'], r['precision']) == (rule, 'fp16')) - next(
                r['nll'] for r in records if (r['rule'], r['precision']) == (rule, 'fp32'))
            comparisons[rule] = {'gradient': comparison, 'fp16_minus_fp32_nll': loss_difference,
                                 'passes_small_cpu_screen': comparison['passes_two_percent_gate']
                                 and abs(loss_difference) <= .003}
        equal_forward = {label: hashes['last_one', label] == hashes['all_sampled_steps', label]
                         for label in ['fp32', 'fp16']}
        if not all(equal_forward.values()):
            raise AssertionError('Gradient rule unexpectedly changed the numerical forward')
        if any(file_sha256(f) != sha for f, sha in sources.items()):
            raise AssertionError('Frozen numerical screen source changed')
        passed = all(r['passes_small_cpu_screen'] for r in comparisons.values())
        result = {'status': 'PASS_SMALL_CPU_FULL_GRADIENT_PRECISION_SCREEN' if passed
                  else 'FAIL_SMALL_CPU_FULL_GRADIENT_PRECISION_SCREEN', 'comparisons': comparisons,
                  'same_forward_hashes_by_precision': equal_forward,
                  'truncated_vs_full_fp32_gradient': gradient_comparison(
                      vectors['all_sampled_steps', 'fp32'], vectors['last_one', 'fp32']),
                  'unique_train_windows': 8, 'unique_real_targets': total,
                  'diagnostic_real_target_presentations': 4*total,
                  'diagnostic_training_matmul_flops': diagnostic_work,
                  'complete_state_readback_exact': True, 'native_qualification_established': False,
                  'scientific_followup_launched': False}
    except BaseException as error:
        result['error'] = str(error); raise
    finally:
        result.update(records=records, wall_seconds=time.time()-started,
                      assigned_chips=0, optimizer_updates=0, test_accessed=False, scope=plan['scope'])
        write_state(out / 'report.json', result); write_state(public / 'report.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
