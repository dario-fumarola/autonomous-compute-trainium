"""CPU-only prospective randomness-policy audit after the frozen pilot finishes."""
import argparse
import concurrent.futures as cf
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from compute_trainium.metrics import DocumentScore, paired_group_interval
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    out = args.root / 'low-depth-prefix-noise-audit-v1'
    public = Path('results/low_depth_prefix_noise_audit_20261025')
    experiment = args.root / 'parcae-low-depth-pilot-v1'
    main_public = Path('results/parcae_low_depth_pilot_20261024/main')
    if out.exists() or (main_public / 'scores-progress.json').exists():
        raise FileExistsError('Register once before any main model scores; preserve failed executions')
    tune = json.loads(Path('results/parcae_low_depth_pilot_20261024/tuning-v2/report.json').read_text())
    if tune['status'] != 'COMPLETE_FROZEN_LOW_DEPTH_LR_SCREEN':
        raise ValueError('Completed rate selection required')
    files = ['scripts/run_low_depth_noise_audit.py', 'scripts/evaluate_low_depth_prefix_noise.py',
             'compute_trainium/prefix_noise_policy.py', 'tests/test_prefix_noise_policy.py',
             'scripts/evaluate_low_depth_cpu.py', 'scripts/prepare_low_depth_main.py',
             'scripts/run_low_depth_main.py', 'compute_trainium/metrics.py']
    started = time.time()
    queries = [{'arm': arm, 'readout': readout, 'depth': depth}
               for readout in ['matched', 'equal']
               for arm in ['fixed1', 'poisson2']
               for depth in ([1] if arm == 'fixed1' else [1, 2, 3])]
    plan = {'identity': 'low-depth-prefix-noise-audit-v1', 'registered_unix': started,
            'deadline_unix': started + 16*3600, 'cpu_seconds_limit': 1200, 'cpu_parallelism': 2,
            'source_sha256': {f: file_sha256(f) for f in files},
            'preparation_plan_sha256': file_sha256(experiment / 'preparation-plan.json'),
            'tuning_plan_sha256': file_sha256(experiment / 'tuning-plan-v2.json'),
            'queries': queries, 'assigned_chips': 0, 'optimizer_updates': 0, 'test_accessed': False,
            'selection_policy': 'No model,rate,exit,seed or endpoint selection from these outcomes.',
            'scope': 'Supplementary same-checkpoint consumed-DEV audit. Frozen primary scoring '
            'uses a whole-document content-hash-derived random-state key,which is not available '
            'before prediction. This audit ignores document identity and keys only the original '
            'window position with the unchanged registered model state seed. No demonstrated '
            'decoding of hash information or explanation of the existing quality gaps is assumed. '
            'Preserve the frozen main readout separately; use prefix-available randomness for any '
            'future deployment/confirmation. Eight endpoint forwards after the complete main DAG; '
            'zero training,heldout-reserve access or terminal alpha.'}
    out.mkdir(); public.mkdir(parents=True, exist_ok=False)
    write_state(out / 'plan.json', plan); write_state(public / 'plan.json', plan)
    failed = threading.Event(); lock = threading.Lock(); active = {}; records = []; costs = []
    pool = None

    def frozen():
        if any(file_sha256(f) != sha for f, sha in plan['source_sha256'].items()):
            raise AssertionError('Registered randomness-policy audit source changed')
        if time.time() > plan['deadline_unix'] or failed.is_set():
            raise TimeoutError('Original randomness-policy audit window/paired stop')

    def run(query, main_plan, main_report, main_sha):
        frozen(); begin = time.time(); process = None
        arm, readout, depth = (query[k] for k in ['arm', 'readout', 'depth'])
        key = f'{arm}-{readout}-d{depth}'; target = out / 'cpu' / key
        original = next(r for r in main_report['scores']
                        if (r['arm'], r['readout'], r['depth']) == (arm, readout, depth))
        native_id = f'low-depth-main-v1-{arm}-' + ('finish' if readout == 'matched' else 'equal')
        native = next(r for r in main_report['native_records'] if r['identity'] == native_id)
        checkpoint = args.root / 'runs' / native_id / f'checkpoint-u{original["training_update"]}.pt'
        gradient = next(r['independent_cpu'] for r in main_report['gradients']
                        if (r['arm'], r['readout']) == (arm, readout))
        if (gradient['status'] != 'PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT'
                or gradient['checkpoint_sha256'] != original['checkpoint_sha256']
                or file_sha256(checkpoint) != original['checkpoint_sha256']):
            raise AssertionError('Independently qualified same-weight endpoint required')
        try:
            command = [os.sys.executable, '-u', '-m', 'scripts.evaluate_low_depth_prefix_noise',
                       '--root', str(args.root), '--plan', str(experiment / 'main-plan.json'),
                       '--plan-sha', main_sha, '--checkpoint', str(checkpoint),
                       '--checkpoint-sha', original['checkpoint_sha256'], '--case', native['case_id'],
                       '--cohort', 'curve', '--depth', str(depth), '--output', str(target),
                       '--upstream', str(args.root / 'upstream-parcae-69284c1'),
                       '--seconds-limit', str(plan['cpu_seconds_limit'])]
            env = dict(os.environ, USE_TORCH_XLA='0', OMP_NUM_THREADS='12', MKL_NUM_THREADS='12')
            with (out / (key + '.log')).open('x') as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            with lock:
                active[key] = process
            while process.poll() is None:
                frozen()
                if time.time() - begin > plan['cpu_seconds_limit']:
                    raise TimeoutError('Randomness-policy CPU phase cap')
                time.sleep(5)
            if process.returncode:
                raise RuntimeError('Prefix-only scoring failed: ' + key)
            result = json.loads((target / 'report.json').read_text())
            if (result['status'] != 'COMPLETE_PREFIX_ONLY_RANDOMNESS_AUDIT_SCORE'
                    or any(result[k] != original[k] for k in ['checkpoint_sha256', 'training_real_targets',
                                                             'training_update', 'real_targets', 'documents'])):
                raise AssertionError('Randomness-policy audit changed weights or targets')
            old_path = experiment / 'main-execution/scores' / key / 'document-scores.json'
            def documents(path):
                return [DocumentScore(r['domain'], r['id'], r['group'], r['nll_sum'], r['real_targets'])
                        for r in json.loads(path.read_text())['records']]
            interval = paired_group_interval(documents(old_path), documents(target / 'document-scores.json'),
                                             draws=2000, seed=1709)
            row = {**query, 'original_nll': original['metrics']['weighted_nll'],
                   'prefix_only_nll': result['metrics']['weighted_nll'],
                   'original_minus_prefix_only_source_interval': interval,
                   'checkpoint_sha256': original['checkpoint_sha256']}
            with lock:
                records.append(row); write_state(public / 'progress.json', {'records': records})
        except BaseException:
            failed.set(); raise
        finally:
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
            with lock:
                active.pop(key, None)
                costs.append({'identity': key, 'started_unix': begin, 'finished_unix': time.time(),
                              'wall_seconds': time.time()-begin, 'assigned_chips': 0,
                              'exit_code': process.returncode if process else None})

    result = {'status': 'STOPPED_LOW_DEPTH_PREFIX_NOISE_AUDIT'}
    try:
        write_state(public / 'status.json', {'status': 'WAITING_FOR_COMPLETE_FROZEN_MAIN', 'assigned_chips': 0})
        while not (main_public / 'report.json').exists():
            frozen()
            queue_report = args.root / 'low-depth-pilot-queue-v2/report.json'
            if queue_report.exists() and json.loads(queue_report.read_text())['status'] != 'COMPLETE_REGISTERED_LOW_DEPTH_PILOT_QUEUE':
                raise RuntimeError('Main queue ended with failed prerequisite')
            time.sleep(5)
        main_report = json.loads((main_public / 'report.json').read_text())
        main_plan = json.loads((experiment / 'main-plan.json').read_text())
        main_sha = file_sha256(experiment / 'main-plan.json')
        if (main_report['status'] != 'COMPLETE_FROZEN_LOW_DEPTH_MAIN_PILOT'
                or main_plan['preparation_plan_sha256'] != plan['preparation_plan_sha256']
                or main_plan['tuning_plan_sha256'] != plan['tuning_plan_sha256']):
            raise AssertionError('Exact completed registered main pilot required')
        frozen(); write_state(public / 'status.json', {'status': 'RUNNING_CPU_ONLY_PREFIX_NOISE_AUDIT', 'assigned_chips': 0})
        pool = cf.ThreadPoolExecutor(2)
        for future in cf.as_completed([pool.submit(run, q, main_plan, main_report, main_sha) for q in queries]):
            future.result()
        if len(records) != 8:
            raise AssertionError('Every registered endpoint audit must finish')
        contrasts = {}
        for readout in ['matched', 'equal']:
            by = {(r['arm'], r['depth']): r for r in records if r['readout'] == readout}
            contrasts[readout] = {}
            for policy in ['original', 'prefix_only']:
                contrasts[readout][policy] = {
                    'one_pass_minus_candidate': by['fixed1', 1][policy+'_nll']-by['poisson2', 2][policy+'_nll'],
                    'candidate_depth1_minus_depth2': by['poisson2', 1][policy+'_nll']-by['poisson2', 2][policy+'_nll']}
        result = {'status': 'COMPLETE_LOW_DEPTH_PREFIX_NOISE_AUDIT', 'contrasts': contrasts,
                  'maximum_absolute_panel_difference_nats': max(abs(r['original_nll']-r['prefix_only_nll']) for r in records)}
    except BaseException as error:
        result['error'] = str(error); raise
    finally:
        failed.set()
        for process in list(active.values()):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        result.update(records=records, cpu_costs=costs, wall_seconds=time.time()-started,
                      assigned_chips=0, optimizer_updates=0, test_accessed=False, scope=plan['scope'])
        write_state(out / 'report.json', result); write_state(public / 'report.json', result)
        write_state(public / 'status.json', {'status': result['status'], 'assigned_chips': 0})


if __name__ == '__main__':
    main()
