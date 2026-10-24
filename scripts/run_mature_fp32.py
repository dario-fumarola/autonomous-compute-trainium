"""Bounded one-pair FP32 development continuation; no TEST or silent restart."""
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
import torch

from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint, file_sha256
from scripts.qualify_native import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if a.output.name != 'mature-fp32-v1' or a.output.parent != a.root:
        raise ValueError('Use the registered private execution identity')
    torch.set_num_threads(2)
    a.output.mkdir(exist_ok=True)
    public = Path('results/mature_fp32_20261024')
    public.mkdir(exist_ok=True)
    started = time.time()
    cutoff = dt.datetime(2026, 10, 24, 9, 40, tzinfo=dt.timezone.utc).timestamp()
    deadline = dt.datetime(2026, 10, 24, 10, 30, tzinfo=dt.timezone.utc).timestamp()
    prefix = np.load(a.root/'natural-append-6500m-v2/packed/prefix_counts.npy', mmap_mode='r')
    def count(update): return int(prefix[update*256].sum())
    q = json.loads(Path('results/trained_gradient_diagnosis_20261024/controls/fp32-final-qualification-report.json').read_text())
    sensitivity = json.loads(Path('results/trained_gradient_diagnosis_20261024/controls/fp32-recovery-nll-report.json').read_text())
    if q['status'] != 'COMPLETE_FP32_GRADIENT_UPDATE_AND_PAIRED_THROUGHPUT_QUALIFICATION' or not sensitivity['within_registered_nll_diagnostic_limit']:
        raise ValueError('Registered FP32 numerical/performance prerequisites missing')
    if json.loads((a.root/'natural-append-6500m-v2/independent-audit.json').read_text())['status'] != 'PASS_INDEPENDENT_SECOND_EXTENSION_AUDIT':
        raise ValueError('Natural source audit missing')
    rates = q['warm_targets_per_second']
    warm = max((count(11520)-count(7680))/rates[arm] for arm in ['ordinary', 'fixed'])
    budget = (warm+3*195+3*100)*1.3+1800
    sources = [str(f) for f in Path('compute_trainium').glob('*.py')]
    sources += ['scripts/'+f for f in ['run_mature_fp32.py', 'train_primitive_fp32.py',
        'train_variable_depth.py', 'evaluate_primitive_fp32.py', 'evaluate_depth_curve.py',
        'qualify_native.py', 'audit_variable_depth.py', 'supervise.py', 'run_native.sh']]
    sources += ['docs/MATURE_FP32_PROTOCOL_20261024.md', 'docs/FP32_NUMERICAL_QUALIFICATION_20261024.md']
    if (a.output/'plan.json').exists():
        plan = json.loads((a.output/'plan.json').read_text())
    else:
        copies = []
        for arm in ['ordinary', 'fixed']:
            original = a.root/'runs'/f'mature-fixed-v1-{arm}-s103-u7680'/'checkpoint.pt'
            saved = torch.load(original, map_location='cpu', mmap=True, weights_only=False)
            if saved['update'] != 7680 or saved['real_tokens'] != count(7680):
                raise ValueError('Source counters changed')
            before = digest({'model': saved['model'], 'optimizer': saved['optimizer'], 'rng': saved['cpu_rng']})
            if digest({'model': saved['model'], 'optimizer': saved['optimizer']}) != saved['state_digest']:
                raise AssertionError('Parent complete state digest differs')
            saved['signature'] = dict(saved['signature'], precision='fp32', batch=1, accumulation=8)
            target = a.output/(arm+'-source.pt')
            if target.exists(): raise FileExistsError(target)
            atomic_checkpoint(target, saved)
            restored = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
            if digest({'model': restored['model'], 'optimizer': restored['optimizer'], 'rng': restored['cpu_rng']}) != before:
                raise AssertionError('Execution metadata copy changed tensors')
            for key in ['cursor', 'real_tokens', 'update', 'schedule', 'training_matmul_flops', 'signature']:
                if restored[key] != saved[key]: raise AssertionError('Copy changed '+key)
            copies.append({'arm': arm, 'parent_sha256': file_sha256(original), 'copy_sha256': file_sha256(target),
                'model_optimizer_rng_digest': before, 'initial_model_digest': saved['initial_model_digest'],
                'only_changed_signature_fields': ['precision', 'batch', 'accumulation']})
            del saved, restored
        if len({r['initial_model_digest'] for r in copies}) != 1:
            raise AssertionError('Original initial weights were not matched')
        plan = {'identity': 'mature-fp32-v1', 'registered_unix': time.time(), 'seed': 103,
            'source_sha256': {f: file_sha256(f) for f in sources}, 'copies': copies,
            'corpus_manifest_sha256': file_sha256(a.root/'natural-append-6500m-v2/manifest.json'),
            'dev_sha256': file_sha256(a.root/'variable-depth-fresh-v1/dev.json'),
            'targets': {str(u): count(u) for u in [7680, 8448, 10368, 11520]},
            'step_flops': {'ordinary': 286044821913600, 'fixed': 483956914913280},
            'budget': {'measured_paired_targets_per_second': rates, 'warm_bottleneck_seconds': warm,
                'with_launch_eval_30_percent_and_cpu_tail_seconds': budget},
            'training_cutoff_unix': cutoff, 'hard_deadline_unix': deadline,
            'scope': 'One existing paired seed, BF16 history below2B then qualified FP32. DEV only. No new confirmation, exact future-bit or compute-efficiency claim.'}
        write_state(a.output/'plan.json', plan)
        write_state(public/'plan.json', plan)
    def frozen():
        if any(file_sha256(f) != h for f, h in plan['source_sha256'].items()):
            raise AssertionError('Saved scientific source changed')
    frozen()
    for row in plan['copies']:
        if file_sha256(a.output/(row['arm']+'-source.pt')) != row['copy_sha256']:
            raise AssertionError('Source copy changed')
    if file_sha256(a.root/'variable-depth-fresh-v1/dev.json') != plan['dev_sha256'] or file_sha256(a.root/'natural-append-6500m-v2/manifest.json') != plan['corpus_manifest_sha256']:
        raise AssertionError('Data/panel changed')
    if time.time()+plan['budget']['with_launch_eval_30_percent_and_cpu_tail_seconds'] > deadline:
        raise TimeoutError('Measured full execution no longer fits hard budget')
    inventory = [{'identity': f'mature-fp32-v1-{arm}-s103-u{u}', 'arm': arm, 'seed': 103,
        'endpoint': '3B' if u == 11520 else 'stable', 'parameters': 150436608}
        for u in [8448, 10368, 11520] for arm in ['ordinary', 'fixed']]
    write_state(a.output/'models.json', inventory)
    write_state(public/'models.json', inventory)
    if a.prepare_only:
        write_state(a.output/'validation.json', {'status': 'PASS_FROZEN_PLAN_COPY_AND_BUDGET_NO_NATIVE_LAUNCH', 'unix': time.time()})
        return
    if (a.output/'report.json').exists(): raise FileExistsError('Closed execution; use a new recovery identity')
    lock = threading.Lock()
    stop = threading.Event()
    active = []
    costs = []
    audit = None
    def abort_native():
        with lock:
            if not stop.is_set():
                stop.set()
                if any(p.poll() is None for p in active):
                    subprocess.run(['docker', 'stop', '--time', '5', 'autonomous-compute-neuron233'], capture_output=True, timeout=25)
    def watchdog():
        while not (a.output/'report.json').exists():
            write_state(a.output/'heartbeat.json', {'unix': time.time(), 'audit_alive': audit.poll() is None})
            if audit.poll() is not None and audit.returncode: abort_native(); return
            time.sleep(10)
    def native(name, module, args, half, evaluation=False):
        frozen()
        if stop.is_set() or (audit.poll() is not None and audit.returncode): raise RuntimeError('Independent audit or peer failed')
        if (a.root/'runs'/name).exists(): raise FileExistsError('No silent stage rerun: '+name)
        limit = deadline if evaluation else cutoff
        if time.time()+120 > limit: raise TimeoutError('Registered phase time exhausted')
        cores = list(range(half*32, half*32+32))
        if evaluation: cores = cores[::4]
        env = dict(os.environ, RESEARCH_ROOT=str(a.root), RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
            RESEARCH_RUN=name, RESEARCH_ENTRY_MODULE=module, RESEARCH_DDP_RANKS=str(len(cores)),
            NEURON_RT_VISIBLE_CORES=','.join(map(str, cores)), RESEARCH_TIMEOUT_SECONDS=str(min(5400, int(limit-time.time()))))
        with (a.output/(name+'.log')).open('x') as log:
            proc = subprocess.Popen([os.sys.executable, '-u', '-m', 'scripts.supervise', *args], env=env, stdout=log, stderr=subprocess.STDOUT)
        with lock: active.append(proc)
        write_state(a.output/(name+'-process.json'), {'pid': proc.pid, 'started_unix': time.time()})
        try:
            code = proc.wait(timeout=max(1, limit-time.time()+30))
        finally:
            if proc.poll() is not None:
                with lock: active.remove(proc)
        closure = json.loads((a.root/'snapshots'/name/'closure.json').read_text())
        with lock:
            costs.append({'identity': name, **closure})
            write_state(public/'execution-costs.json', {'status': 'RUNNING', 'attempts': costs,
                'scope': 'Overlapping native assignments; CPU audit and preparation have separate intervals. Qualification is excluded from scientific exposure.'})
        if code: raise RuntimeError('Native stage failed: '+name)
        frozen()
        return json.loads((a.root/'runs'/name/'report.json').read_text())
    def fit(arm, half):
        parent = '/work/instance-nvme/mature-fp32-v1/'+arm+'-source.pt'
        prior = 7680
        for target in [8448, 10368, 11520]:
            name = f'mature-fp32-v1-{arm}-s103-u{target}'
            base = json.loads((a.root/'snapshots'/f'mature-fixed-v1-{arm}-s103-u7680'/'launch.json').read_text())['arguments']
            args = base+['--resume', parent, '--purpose', 'scientific', '--precision', 'fp32', '--batch', '1',
                '--accumulation', '8', '--steps', str(target-prior), '--warmup', '4', '--decay-start', str(count(10368)),
                '--decay-end', str(count(11520)), '--checkpoint-every', '1024']
            row = native(name, 'scripts.train_primitive_fp32', args, half)
            if not row['all_replica_model_optimizer_states_equal'] or row['total_real_targets'] != count(target) or row['total_training_matmul_flops'] != target*plan['step_flops'][arm]:
                raise AssertionError('Scientific phase state/counters differ')
            parent = '/work/instance-nvme/runs/'+name+'/checkpoint.pt'
            prior = target
            depth = 1 if arm == 'ordinary' else 2
            result = native(name+f'-dev-r{depth}', 'scripts.evaluate_primitive_fp32', ['--checkpoint', parent,
                '--corpus', '/work/instance-nvme/variable-depth-fresh-v1/dev.json', '--split', 'dev', '--cohort', 'curve',
                '--repetitions', str(depth)], half, True)
            with lock:
                progress = json.loads((public/'progress.json').read_text()) if (public/'progress.json').exists() else {'rows': [], 'scope': plan['scope']}
                progress['rows'].append({'arm': arm, 'seed': 103, 'update': target, 'real_targets': count(target),
                    'phase': 'annealed' if target == 11520 else 'stable', 'native_fp32_metrics': result['metrics'],
                    'independent_cpu_audit': 'pending; primary once completed'})
                write_state(public/'progress.json', progress)
    try:
        with (a.output/'independent-cpu.log').open('x') as log:
            audit = subprocess.Popen([os.sys.executable, '-u', '-m', 'scripts.audit_variable_depth', '--root', str(a.root),
                '--models', str(a.output/'models.json'), '--corpus', str(a.root/'variable-depth-fresh-v1/dev.json'),
                '--output', str(a.output/'independent-cpu'), '--deadline-unix', str(deadline-60)],
                env=dict(os.environ, USE_TORCH_XLA='0', OMP_NUM_THREADS='12', MKL_NUM_THREADS='12'), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_state(a.output/'independent-cpu-process.json', {'pid': audit.pid, 'started_unix': time.time()})
        threading.Thread(target=watchdog, daemon=True).start()
        with cf.ThreadPoolExecutor(2) as pool:
            tasks = [pool.submit(fit, arm, half) for half, arm in enumerate(['ordinary', 'fixed'])]
            for task in cf.as_completed(tasks):
                try: task.result()
                except BaseException: abort_native(); raise
        audit.wait(timeout=max(1, deadline-time.time()-30))
        if audit.returncode: raise RuntimeError('Independent full-panel audit failed')
        frozen()
        write_state(public/'independent-cpu-verification.json', json.loads((a.output/'independent-cpu/report.json').read_text()))
        report = {'status': 'COMPLETE_ONE_PAIR_FP32_3B_DEV_DIAGNOSTIC', 'wall_seconds': time.time()-started,
            'test_accessed': False, 'new_terminal_attempt': False, 'scope': plan['scope']}
    except BaseException as error:
        abort_native()
        if audit and audit.poll() is None:
            os.killpg(audit.pid, signal.SIGTERM)
            try: audit.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(audit.pid, signal.SIGKILL); audit.wait(timeout=10)
        report = {'status': 'STOPPED_ONE_PAIR_FP32_3B_DEV_DIAGNOSTIC', 'error': str(error),
            'wall_seconds': time.time()-started, 'test_accessed': False}
        raise
    finally:
        write_state(a.output/'report.json', report)
        write_state(public/'report.json', report)


if __name__ == '__main__':
    main()
