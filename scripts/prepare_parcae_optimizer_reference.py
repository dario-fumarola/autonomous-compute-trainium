"""Actual-shape independent CPU optimizer vectors, no new learning or text."""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import torch

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_candidate import ParcaeOptimizerCandidate
from compute_trainium.parcae_update_qualification import (
    UPDATE_FIXTURE, assign_gradient, compare_moments, compare_update,
    configure_step, named_moments,
)
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint, file_sha256
from scripts.qualify_parcae_optimizer_cpu import unwrapped_reference
from third_party.parcae_reference import muon_adamw as reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--attempt', choices=['attempt2'], required=True)
    args = parser.parse_args()
    out = args.root / 'parcae-optimizer-global-v1' / args.attempt
    out.mkdir(parents=True, exist_ok=False)
    public = Path('results/parcae_optimizer_global_20261024') / args.attempt
    public.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(2)
    unwrapped_reference()
    reference_root = args.root / 'parcae-native-global-v1'
    proof = json.loads((reference_root / 'report.json').read_text())
    if proof['status'] != 'COMPLETE_INDEPENDENT_PARCAE_CPU_GLOBAL_REFERENCES':
        raise ValueError('Independent natural TRAIN gradients are incomplete')
    files = ['scripts/prepare_parcae_optimizer_reference.py',
        'compute_trainium/parcae_update_qualification.py',
        'compute_trainium/parcae_optimizer_candidate.py',
        'third_party/parcae_reference/muon_adamw.py']
    plan = {'identity': 'parcae-optimizer-global-v1', 'attempt': args.attempt, 'registered_unix': started,
        'source_sha256': {p: file_sha256(p) for p in files},
        'cpu_global_report_sha256': file_sha256(reference_root / 'report.json'),
        'gate_policy': 'release_muon', 'updates': UPDATE_FIXTURE,
        'cpu_update_relative_gate': 2e-5, 'cpu_moment_relative_gate': 2e-6,
        'cpu_seconds_limit': 1200, 'test_accessed': False,
        'scope': 'Actual-shape optimizer numerical fixtures derived from already qualified TRAIN gradient vectors. No model forward, corpus generation, held-out scoring or research training.'}
    write_state(out / 'plan.json', plan)
    write_state(public / 'preparation-plan.json', plan)
    records = []

    def guard():
        if time.time() - started > 1200:
            raise TimeoutError('CPU optimizer reference twenty-minute limit')
        available = next(int(line.split()[1]) * 1024 for line in
            Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if available < 544 * 1024**3:
            raise MemoryError('CPU optimizer preparation memory reserve')
        if shutil.disk_usage(out).free < 42 * 1024**3 or shutil.disk_usage('/').free < 35 * 1024**3:
            raise OSError('CPU optimizer preparation disk reserve')

    try:
        for row in proof['records']:
            guard()
            arm = row['arm']
            begin = time.time()
            target = out / arm
            target.mkdir()
            checkpoint = reference_root / arm / 'qualification-state.pt'
            gradient_file = reference_root / arm / 'global-cpu-fp32.pt'
            if file_sha256(checkpoint) != row['checkpoint_sha256'] or file_sha256(gradient_file) != row['global_gradient_file_sha256']:
                raise AssertionError('Prescribed weights or natural TRAIN gradients changed')
            saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
            source = torch.load(gradient_file, map_location='cpu', mmap=True, weights_only=False)
            if not saved['qualification_only']:
                raise ValueError('Qualification-only initial state required')
            config = ParcaeCandidateConfig(**saved['config'])
            model = ParcaeCandidate(config, recurrent=saved['recurrent'], initialize=False)
            other = ParcaeCandidate(config, recurrent=saved['recurrent'], initialize=False)
            model.load_state_dict(saved['model']); other.load_state_dict(saved['model'])
            expected = reference.MuonAdamW(reference.get_muon_param_groups(
                model.named_parameters(), adamw_lr=.008, adamw_betas=(.8, .95),
                adamw_eps=1e-10, adamw_wd=0., muon_lr=.008))
            actual = ParcaeOptimizerCandidate(other, gate_policy='release_muon', fused_adamw=True)
            lookup = {id(p): n for n, p in model.named_parameters()}
            if [[lookup[id(p)] for p in g['params']] for g in expected.param_groups] != actual.recipe['groups']:
                raise AssertionError('Actual-shape parameter grouping differs')
            errors = []
            for update in range(len(UPDATE_FIXTURE)):
                guard()
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                actual_before = {n: p.detach().clone() for n, p in other.named_parameters()}
                for candidate, opt in [(model, expected), (other, actual)]:
                    scale = configure_step(opt, update)
                    assign_gradient(candidate, source['gradients'], scale)
                    opt.step()
                weights = {n: p.detach().clone() for n, p in model.named_parameters()}
                actual_weights = {n: p.detach().clone() for n, p in other.named_parameters()}
                moments = named_moments(model, expected, reference=True)
                delta = compare_update(before, weights, actual_before, actual_weights)
                state_delta = compare_moments(moments, named_moments(other, actual))
                if delta['relative_l2'] > 2e-5 or state_delta['relative_l2'] > 2e-6:
                    raise AssertionError('Actual-shape CPU release update/moment disagreement')
                file = target / f'cpu-step-{update+1}.pt'
                atomic_checkpoint(file, {'model': weights, 'moments': moments,
                    'update': update+1, 'qualification_only': True})
                errors.append({'update': update+1, 'update_comparison': delta,
                    'moment_comparison': state_delta, 'checkpoint_sha256': file_sha256(file)})
                write_state(public / 'cpu-progress.json', {'completed_arms': records,
                    'active_arm': arm, 'completed_update_vectors': errors})
                del before, actual_before, weights, actual_weights, moments
                gc.collect()
            records.append({'arm': arm, 'parameters': config.parameter_count(saved['recurrent']),
                'initial_checkpoint_sha256': row['checkpoint_sha256'],
                'prescribed_gradient_sha256': row['global_gradient_file_sha256'],
                'vectors': errors, 'active_seconds': time.time()-begin})
            del model, other, expected, actual, saved, source
            gc.collect()
        if any(file_sha256(p) != sha for p, sha in plan['source_sha256'].items()):
            raise AssertionError('CPU optimizer preparation source changed')
        report = {'status': 'PASS_ACTUAL_SHAPE_CPU_RELEASE_OPTIMIZER_VECTORS',
            'records': records, 'assigned_chips': 0, 'test_accessed': False,
            'prescribed_software_updates_per_implementation': len(UPDATE_FIXTURE),
            'research_optimizer_updates': 0, 'wall_seconds_after_imports': time.time()-started,
            'scope': plan['scope']}
    except BaseException as error:
        report = {'status': 'STOPPED_ACTUAL_SHAPE_CPU_OPTIMIZER_PREPARATION',
            'error': str(error), 'records': records, 'assigned_chips': 0,
            'research_optimizer_updates': 0, 'wall_seconds_after_imports': time.time()-started}
        raise
    finally:
        write_state(out / 'report.json', report)
        write_state(public / 'cpu-report.json', report)


if __name__ == '__main__':
    main()
