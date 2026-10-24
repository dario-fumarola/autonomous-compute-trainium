"""Bounded, independently tuned ordinary-versus-cheap architectural control."""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    identity = 'reinjection-pilot-v1'
    output = a.root/identity
    public = Path('results/reinjection_pilot_20261024')
    output.mkdir(exist_ok=True)
    public.mkdir(exist_ok=True)
    started = time.time()
    q = json.loads((a.root/'reinjection-native-q-v1/report.json').read_text())
    if q['status'] != 'PASS_REINJECTION_NATIVE_QUALIFICATION':
        raise ValueError('New control has not passed native qualification')
    old = json.loads(Path('results/mature_fp32_20261024/plan.json').read_text())
    rate = min(q['measured_warm_targets_per_second'],
               old['budget']['measured_paired_targets_per_second']['ordinary'])
    prefix = np.load(a.root/'natural-append-6500m-v2/packed/prefix_counts.npy', mmap_mode='r')
    def count(u): return int(prefix[u*256].sum())
    cutoff = dt.datetime(2026, 10, 24, 18, tzinfo=dt.timezone.utc).timestamp()
    deadline = dt.datetime(2026, 10, 24, 19, tzinfo=dt.timezone.utc).timestamp()
    # Eight tuning fits share half1; two main fits can overlap across halves.
    tuning = 8*(count(1152)/rate+195+100)+1200
    main_cost = count(3840)/rate+3*(195+100)
    budget = (tuning+main_cost)*1.3+1800
    if not (output/'plan.json').exists():
        hashes = dict(old['source_sha256'])
        files = ['compute_trainium/reinjection_control.py', 'scripts/train_reinjection_control.py',
                 'scripts/evaluate_reinjection_control.py', 'scripts/audit_reinjection_pilot.py',
                 'scripts/run_reinjection_pilot.py', 'docs/REINJECTION_PILOT_PROTOCOL_20261024.md']
        hashes.update({f: file_sha256(f) for f in files})
        plan = {'identity': identity, 'registered_unix': time.time(), 'source_sha256': hashes,
                'dev_sha256': old['dev_sha256'], 'corpus_manifest_sha256': old['corpus_manifest_sha256'],
                'native_qualification_sha256': file_sha256(a.root/'reinjection-native-q-v1/report.json'),
                'lr_grid': [.0003, .0006, .001, .0015], 'tuning_seed': 11, 'main_seed': 419,
                'updates': [384, 1037, 1152, 3456, 3840],
                'targets': {str(u): count(u) for u in [384, 1037, 1152, 3456, 3840]},
                'measured_conservative_targets_per_second': rate,
                'budget_seconds': budget, 'training_cutoff_unix': cutoff,
                'hard_deadline_unix': deadline, 'futility_tune_penalty': .02,
                'scope': 'One fresh development pair, existing consumed TUNE/DEV. Same parameters/source/estimated matmuls. No TEST or terminal alpha; an architectural connection control, not a latent-depth win.'}
        write_state(output/'plan.json', plan)
        write_state(public/'plan.json', plan)
    plan = json.loads((output/'plan.json').read_text())
    def frozen():
        if any(file_sha256(f) != h for f, h in plan['source_sha256'].items()):
            raise AssertionError('Frozen pilot source changed')
        if (file_sha256(a.root/'variable-depth-fresh-v1/dev.json') != plan['dev_sha256']
                or file_sha256(a.root/'natural-append-6500m-v2/manifest.json') != plan['corpus_manifest_sha256']):
            raise AssertionError('Source/panel changed')
    frozen()
    if time.time()+plan['budget_seconds'] > deadline or time.time()+plan['budget_seconds']-1800 > cutoff:
        raise TimeoutError('Full measured pilot budget no longer fits')
    if a.prepare_only: return
    if (output/'report.json').exists(): raise FileExistsError('Closed pilot; no silent restart')
    active, cpu, costs = [], [], []
    lock = threading.Lock()
    stop = threading.Event()
    report = None
    def abort():
        stop.set()
        if any(p.poll() is None for p in active):
            subprocess.run(['docker', 'stop', '--time', '5', 'autonomous-compute-neuron233'],
                           capture_output=True, timeout=25)
    def native(name, module, args, half, evaluation=False, expected_new_targets=0):
        frozen()
        if stop.is_set() or any(p.poll() is not None and p.returncode for p in cpu):
            raise RuntimeError('Peer or independent auditor failed')
        if (a.root/'runs'/name).exists(): raise FileExistsError(name)
        limit = deadline if evaluation else cutoff
        estimate = 120 if evaluation else expected_new_targets/rate*1.3+195
        if time.time()+estimate > limit: raise TimeoutError('Full phase no longer fits')
        cores = list(range(half*32, half*32+32, 4 if evaluation else 1))
        env = dict(os.environ, RESEARCH_ROOT=str(a.root), RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
            RESEARCH_RUN=name, RESEARCH_ENTRY_MODULE=module, RESEARCH_DDP_RANKS=str(len(cores)),
            NEURON_RT_VISIBLE_CORES=','.join(map(str, cores)),
            RESEARCH_TIMEOUT_SECONDS=str(min(7200, int(limit-time.time()))))
        with (output/(name+'.log')).open('x') as log:
            proc = subprocess.Popen([os.sys.executable, '-u', '-m', 'scripts.supervise', *args],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        with lock: active.append(proc)
        write_state(output/(name+'-process.json'), {'pid': proc.pid, 'started_unix': time.time()})
        while proc.poll() is None:
            if time.time() > limit or stop.is_set() or any(p.poll() is not None and p.returncode for p in cpu):
                abort()
                raise RuntimeError('Native deadline, peer or independent audit failure')
            with lock: write_state(output/'heartbeat.json', {'unix': time.time(), 'last_observed_active_phase': name})
            time.sleep(5)
        closure = json.loads((a.root/'snapshots'/name/'closure.json').read_text())
        with lock:
            active.remove(proc)
            costs.append({'identity': name, **closure})
            write_state(public/'execution-costs.json', {'status': 'RUNNING', 'attempts': costs})
        if proc.returncode:
            abort()
            raise RuntimeError('Native phase failed: '+name)
        frozen()
        return json.loads((a.root/'runs'/name/'report.json').read_text())
    def arguments(lr, seed, target, tuning=False):
        end = 1152 if tuning else 3840
        decay = 1037 if tuning else 3456
        return ['--corpus', '/work/instance-nvme/natural-append-6500m-v2/source.json',
            '--packed-corpus', '/work/instance-nvme/natural-append-6500m-v2/packed',
            '--dev-corpus', '/work/instance-nvme/variable-depth-fresh-v1/dev.json',
            '--architecture', 'preln', '--repetitions', '1', '--depth-policy', 'ordinary',
            '--precision', 'fp32', '--context', '1024', '--batch', '1', '--accumulation', '8',
            '--optimizer', 'fused', '--recipe', 'pretraining', '--seed', str(seed), '--data-seed', '23',
            '--lr', str(lr), '--lr-warmup-tokens', str(int(count(end)*.025)),
            '--decay-start', str(count(decay)), '--decay-end', str(count(end)),
            '--steps', str(target), '--warmup', '4', '--reference-threads', '8',
            '--purpose', 'tuning' if tuning else 'scientific', '--save', '--checkpoint-every', '1024']
    def evaluate(name, arm, half, cohort):
        checkpoint = '/work/instance-nvme/runs/'+name+'/checkpoint.pt'
        evaluation = name+'-'+cohort+'-r1'
        module = 'scripts.evaluate_reinjection_control' if arm == 'reinjection' else 'scripts.evaluate_primitive_fp32'
        row = native(evaluation, module, ['--checkpoint', checkpoint,
            '--corpus', '/work/instance-nvme/variable-depth-fresh-v1/dev.json', '--split', 'dev',
            '--cohort', cohort, '--repetitions', '1'], half, True)
        return evaluation, row
    def auditor(models, name, cohort):
        path = output/(name+'-models.json')
        write_state(path, models)
        with (output/(name+'.log')).open('x') as log:
            proc = subprocess.Popen([os.sys.executable, '-u', '-m', 'scripts.audit_reinjection_pilot',
                '--root', str(a.root), '--models', str(path), '--corpus', str(a.root/'variable-depth-fresh-v1/dev.json'),
                '--output', str(output/name), '--deadline-unix', str(deadline-60), '--cohort', cohort],
                env=dict(os.environ, USE_TORCH_XLA='0', OMP_NUM_THREADS='12', MKL_NUM_THREADS='12'),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        cpu.append(proc)
        write_state(output/(name+'-process.json'), {'pid': proc.pid, 'started_unix': time.time()})
        return proc
    try:
        tuning_rows = []
        pending = [(arm, i, lr) for arm in ['ordinary', 'reinjection']
                   for i, lr in enumerate(plan['lr_grid'])]
        def tune_worker(half):
            if half == 0:
                owner = a.root/'mature-fp32-compute-v1/report.json'
                while not owner.exists():
                    with lock:
                        if not pending: return
                    if stop.is_set() or time.time()+count(1152)/rate*1.3+295 > cutoff:
                        raise TimeoutError('Tuning peer/core-owner deadline')
                    time.sleep(10)
                if json.loads(owner.read_text())['status'] != 'COMPLETE_ONE_PAIR_FP32_3B_COMPUTE_CONTROL':
                    raise RuntimeError('Prior ordinary core owner stopped')
            while True:
                with lock:
                    if not pending: return
                    arm, i, lr = pending.pop(0)
                module = 'scripts.train_reinjection_control' if arm == 'reinjection' else 'scripts.train_primitive_fp32'
                name = f'{identity}-tune-{arm}-lr{i}-s11'
                row = native(name, module, arguments(lr, 11, 1152, True), half, expected_new_targets=count(1152))
                if (not row['all_replica_model_optimizer_states_equal'] or row['parameters'] != 150436608
                        or row['total_real_targets'] != count(1152)):
                    raise AssertionError('Tuning counters/state/parameter count differ')
                evaluation, score = evaluate(name, arm, half, 'tune')
                with lock:
                    tuning_rows.append({'identity': name, 'arm': arm, 'seed': 11, 'endpoint': 'tune301M',
                        'evaluation_identity': evaluation, 'lr': lr, 'metrics': score['metrics'],
                        'initial_model_digest': row['initial_model_digest']})
                    write_state(public/'tuning.json', {'rows': tuning_rows})
        with cf.ThreadPoolExecutor(2) as pool:
            tasks = [pool.submit(tune_worker, half) for half in [1, 0]]
            for task in cf.as_completed(tasks):
                try: task.result()
                except BaseException: abort(); raise
        if len({r['initial_model_digest'] for r in tuning_rows}) != 1:
            raise AssertionError('TUNE arms did not have matched initial weights')
        chosen = {}
        for arm in ['ordinary', 'reinjection']:
            subset = [r for r in tuning_rows if r['arm'] == arm]
            best = min(r['metrics']['weighted_nll'] for r in subset)
            chosen[arm] = min((r for r in subset if r['metrics']['weighted_nll'] <= best+.0002), key=lambda r: r['lr'])
        write_state(public/'selected.json', chosen)
        audit = auditor(list(chosen.values()), 'tune-independent-cpu', 'tune')
        audit.wait(timeout=max(1, min(2100, deadline-time.time()-60)))
        if audit.returncode: raise RuntimeError('Selected TUNE independent audit failed')
        audited = json.loads((output/'tune-independent-cpu/report.json').read_text())
        write_state(public/'tune-independent-cpu.json', audited)
        scores = {r['arm']: r['metrics']['weighted_nll'] for r in audited['records']}
        if scores['reinjection']-scores['ordinary'] > plan['futility_tune_penalty']:
            report = {'status': 'STOPPED_AT_PREDECLARED_TUNE_FUTILITY', 'selected_cpu_nll': scores,
                      'test_accessed': False, 'wall_seconds': time.time()-started, 'scope': plan['scope']}
        else:
            models = [{'identity': f'{identity}-{arm}-s419-u{u}', 'arm': arm, 'seed': 419,
                'endpoint': '1B' if u == 3840 else 'stable',
                'evaluation_identity': f'{identity}-{arm}-s419-u{u}-curve-r1'}
                for u in [384, 1152, 3840] for arm in ['ordinary', 'reinjection']]
            write_state(public/'models.json', models)
            shadow = auditor(models, 'main-independent-cpu', 'curve')
            initial = {}
            def fit(arm, half):
                if half == 0:
                    owner = a.root/'mature-fp32-compute-v1/report.json'
                    while not owner.exists():
                        if stop.is_set() or time.time()+main_cost*1.3 > cutoff:
                            raise TimeoutError('Prior ordinary core owner did not close in time')
                        time.sleep(10)
                    if json.loads(owner.read_text())['status'] != 'COMPLETE_ONE_PAIR_FP32_3B_COMPUTE_CONTROL':
                        raise RuntimeError('Prior core owner stopped')
                parent = None
                prior = 0
                module = 'scripts.train_reinjection_control' if arm == 'reinjection' else 'scripts.train_primitive_fp32'
                for target in [384, 1152, 3840]:
                    name = f'{identity}-{arm}-s419-u{target}'
                    args = arguments(chosen[arm]['lr'], 419, target-prior)
                    if parent: args += ['--resume', parent]
                    row = native(name, module, args, half, expected_new_targets=count(target)-count(prior))
                    if (not row['all_replica_model_optimizer_states_equal'] or row['parameters'] != 150436608
                            or row['total_real_targets'] != count(target)):
                        raise AssertionError('Main state/target/parameter counts differ')
                    with lock:
                        initial[arm] = row['initial_model_digest']
                        if len(set(initial.values())) != 1: raise AssertionError('Main paired initial weights differ')
                    _, score = evaluate(name, arm, half, 'curve')
                    with lock:
                        path = public/'progress.json'
                        progress = json.loads(path.read_text()) if path.exists() else {'rows': []}
                        progress['rows'].append({'arm': arm, 'seed': 419, 'update': target,
                            'real_targets': count(target), 'phase': 'annealed' if target == 3840 else 'stable',
                            'native_metrics': score['metrics']})
                        write_state(path, progress)
                    parent = '/work/instance-nvme/runs/'+name+'/checkpoint.pt'
                    prior = target
            with cf.ThreadPoolExecutor(2) as pool:
                tasks = [pool.submit(fit, arm, half) for half, arm in enumerate(['ordinary', 'reinjection'])]
                for task in cf.as_completed(tasks):
                    try: task.result()
                    except BaseException: abort(); raise
            shadow.wait(timeout=max(1, deadline-time.time()-60))
            if shadow.returncode: raise RuntimeError('Main independent CPU audit failed')
            write_state(public/'independent-cpu-verification.json', json.loads((output/'main-independent-cpu/report.json').read_text()))
            frozen()
            report = {'status': 'COMPLETE_ONE_PAIR_TRAINED_REINJECTION_CONTROL', 'wall_seconds': time.time()-started,
                      'test_accessed': False, 'new_terminal_attempt': False, 'scope': plan['scope']}
    except BaseException as error:
        abort()
        report = {'status': 'STOPPED_REINJECTION_PILOT', 'error': str(error),
                  'wall_seconds': time.time()-started, 'test_accessed': False}
        raise
    finally:
        for proc in cpu:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
        write_state(output/'report.json', report)
        write_state(public/'report.json', report)
        write_state(public/'execution-costs.json', {'status': report['status'], 'attempts': costs,
            'wall_seconds': time.time()-started,
            'scope': 'Tuning, failed attempts, discarded paths and native launches included. CPU audit intervals are separate. Half-node jobs overlap the mature compute control; do not add wall intervals as sequential reservation time.'})


if __name__ == '__main__':
    main()
