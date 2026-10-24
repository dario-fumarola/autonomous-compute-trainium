"""Common-FP32 second-core control with strong source/compute baselines."""
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
from compute_trainium.execution_isolation import cancel_worker_container
from compute_trainium.owned_native_phase import create_worker
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    identity = 'depth-connection-v1'
    out, public = a.root/identity, Path('results/depth_connection_20261024')
    out.mkdir(exist_ok=True)
    public.mkdir(exist_ok=True)
    started = time.time()
    prior = Path('results/reinjection_pilot_20261024')
    if json.loads((prior/'report.json').read_text())['status'] != 'COMPLETE_ONE_PAIR_TRAINED_REINJECTION_CONTROL':
        raise ValueError('Previous common-precision controls incomplete')
    q = json.loads(Path('results/fp16_native_gradient_20261024/report.json').read_text())
    if not q['qualifies_both_mature_global_objectives']:
        raise ValueError('Isolated native runtime qualification incomplete; FP16 is still not adopted')
    old = json.loads(Path('results/mature_fp32_20261024/plan.json').read_text())
    rates = old['budget']['measured_paired_targets_per_second']
    prefix = np.load(a.root/'natural-append-6500m-v2/packed/prefix_counts.npy', mmap_mode='r')
    def count(u): return int(prefix[u*256].sum())
    cutoff = dt.datetime(2026, 10, 24, 23, tzinfo=dt.timezone.utc).timestamp()
    deadline = dt.datetime(2026, 10, 25, 0, tzinfo=dt.timezone.utc).timestamp()
    continuation = (count(6496)-count(1152))/rates['ordinary']+295
    tuning = 4*(count(1152)/rates['fixed']+295)
    fixed_main = count(3840)/rates['fixed']+295
    budget = (max(continuation, tuning)+600+max(continuation, fixed_main))*1.3+1800
    if not (out/'plan.json').exists():
        previous_audits = json.loads((prior/'independent-cpu-verification.json').read_text())['records']
        parents = {}
        for arm in ['ordinary', 'reinjection']:
            name = f'reinjection-pilot-v1-{arm}-s419-u1152'
            proof = next(r for r in previous_audits if r['identity'] == name)
            source = a.root/'runs'/name/'checkpoint.pt'
            if not proof['copy_checksum_and_full_state_readback'] or file_sha256(source) != proof['checkpoint_sha256']:
                raise AssertionError('Stable source has no matching independent state audit')
            parents[arm] = {'identity': name, 'checkpoint_sha256': proof['checkpoint_sha256']}
        reference = json.loads((a.root/'runs/reinjection-pilot-v1-ordinary-s419-u3840/report.json').read_text())
        tune = json.loads((prior/'tuning.json').read_text())['rows']
        if len({r['initial_model_digest'] for r in tune}) != 1:
            raise AssertionError('Prior TUNE initialization mismatch')
        files = ['compute_trainium/reinjection_control.py', 'compute_trainium/execution_isolation.py',
                 'compute_trainium/owned_native_phase.py', 'scripts/run_depth_connection_campaign.py',
                 'scripts/audit_depth_connection.py', 'scripts/train_reinjection_control.py',
                 'scripts/evaluate_reinjection_control.py', 'scripts/supervise_isolated.py',
                 'scripts/run_native_isolated.sh', 'docs/DEPTH_CONNECTION_PROTOCOL_20261024.md']
        hashes = {**old['source_sha256'], **{f: file_sha256(f) for f in files}}
        plan = {'identity': identity, 'registered_unix': time.time(), 'parents': parents,
            'source_sha256': hashes, 'dev_sha256': old['dev_sha256'],
            'corpus_manifest_sha256': old['corpus_manifest_sha256'],
            'initial_main_digest': reference['initial_model_digest'],
            'initial_tune_digest': tune[0]['initial_model_digest'],
            'lr_grid': [.0003, .0006, .001, .0015], 'tune_seed': 11, 'main_seed': 419,
            'targets': {str(u): count(u) for u in [1037, 1152, 3456, 3840, 5846, 6496]},
            'budget_seconds': budget, 'measured_fp32_rates': rates,
            'training_cutoff_unix': cutoff, 'hard_deadline_unix': deadline,
            'scope': 'One-pair common-FP32-from-init second-core development control. Reused consumed TUNE/DEV, strong inexpensive/ordinary matched-work references. No TEST, terminal significance or FP16 training.'}
        write_state(out/'plan.json', plan)
        write_state(public/'plan.json', plan)
    plan = json.loads((out/'plan.json').read_text())
    def frozen():
        if any(file_sha256(f) != h for f, h in plan['source_sha256'].items()):
            raise AssertionError('Frozen scientific source changed')
        if (file_sha256(a.root/'variable-depth-fresh-v1/dev.json') != plan['dev_sha256']
                or file_sha256(a.root/'natural-append-6500m-v2/manifest.json') != plan['corpus_manifest_sha256']):
            raise AssertionError('Panel/source changed')
    frozen()
    if time.time()+plan['budget_seconds'] > deadline or time.time()+plan['budget_seconds']-1800 > cutoff:
        raise TimeoutError('Entire measured scientific window no longer fits')
    if a.prepare_only: return
    if (out/'report.json').exists(): raise FileExistsError('No silent campaign rerun')
    lock = threading.Lock()
    active, cpu, costs = {}, [], []
    report = None
    def native(name, module, args, half, evaluation=False):
        frozen()
        if any(p.poll() is not None and p.returncode for p in cpu):
            raise RuntimeError('Independent audit failed')
        if (a.root/'runs'/name).exists(): raise FileExistsError(name)
        container = create_worker(a.root, name, half)
        with lock: active[name] = container
        limit = deadline if evaluation else cutoff
        env = dict(os.environ, RESEARCH_ROOT=str(a.root), RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
            RESEARCH_RUN=name, RESEARCH_OWNER=name, RESEARCH_CONTAINER=container,
            RESEARCH_ENTRY_MODULE=module, RESEARCH_DDP_RANKS='8' if evaluation else '32',
            NEURON_RT_VISIBLE_CORES=','.join(map(str, range(half*32, half*32+32, 4 if evaluation else 1))),
            RESEARCH_TIMEOUT_SECONDS=str(min(15000, int(limit-time.time()))))
        proc = None
        try:
            with (out/(name+'.log')).open('x') as log:
                proc = subprocess.Popen([os.sys.executable, '-u', '-m', 'scripts.supervise_isolated', *args],
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
            write_state(out/(name+'-process.json'), {'pid': proc.pid, 'started_unix': time.time()})
            while proc.poll() is None:
                if time.time()>limit or any(p.poll() is not None and p.returncode for p in cpu):
                    raise RuntimeError('Deadline or independent CPU audit failure')
                with lock: write_state(out/'heartbeat.json', {'unix': time.time(), 'last_active_phase': name})
                time.sleep(5)
            if proc.returncode: raise RuntimeError('Native phase failed: '+name)
            frozen()
            return json.loads((a.root/'runs'/name/'report.json').read_text())
        finally:
            cancel_worker_container(container, name, 'phase_complete')
            if proc is not None and proc.poll() is None:
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: proc.terminate()
            closure = a.root/'snapshots'/name/'closure.json'
            with lock:
                active.pop(name, None)
                costs.append({'identity': name, **json.loads(closure.read_text())} if closure.exists()
                             else {'identity': name, 'closure_missing': True})
                write_state(public/'execution-costs.json', {'status': 'RUNNING', 'attempts': costs})
    def evaluate(name, arm, half, cohort):
        depth = 2 if arm=='fixed' else 1
        evaluation = name+f'-{cohort}-r{depth}'
        module = 'scripts.evaluate_reinjection_control' if arm=='reinjection' else 'scripts.evaluate_primitive_fp32'
        row = native(evaluation, module, ['--checkpoint', '/work/instance-nvme/runs/'+name+'/checkpoint.pt',
            '--corpus', '/work/instance-nvme/variable-depth-fresh-v1/dev.json', '--split', 'dev',
            '--cohort', cohort, '--repetitions', str(depth)], half, True)
        return evaluation, row
    def audit(models, tag, cohort):
        path = out/(tag+'-models.json')
        write_state(path, models)
        with (out/(tag+'.log')).open('x') as log:
            proc = subprocess.Popen([os.sys.executable, '-u', '-m', 'scripts.audit_depth_connection',
                '--root', str(a.root), '--models', str(path), '--corpus', str(a.root/'variable-depth-fresh-v1/dev.json'),
                '--output', str(out/tag), '--deadline-unix', str(deadline-60), '--cohort', cohort],
                env=dict(os.environ, USE_TORCH_XLA='0', OMP_NUM_THREADS='12', MKL_NUM_THREADS='12'),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        cpu.append(proc)
        write_state(out/(tag+'-process.json'), {'pid': proc.pid, 'started_unix': time.time()})
        return proc
    models = [{'identity': f'{identity}-{arm}-s419-u{3840 if arm=="fixed" else 6496}',
        'arm': arm, 'seed': 419, 'endpoint': '1B' if arm=='fixed' else 'matched',
        'evaluation_identity': f'{identity}-{arm}-s419-u{3840 if arm=="fixed" else 6496}-curve-r{2 if arm=="fixed" else 1}'}
        for arm in ['ordinary', 'reinjection', 'fixed']]
    write_state(public/'models.json', models)
    def continuation(arm, half):
        parent = plan['parents'][arm]['identity']
        if file_sha256(a.root/'runs'/parent/'checkpoint.pt') != plan['parents'][arm]['checkpoint_sha256']:
            raise AssertionError('Audited stable parent changed')
        args = json.loads((a.root/'snapshots'/parent/'launch.json').read_text())['arguments']
        args += ['--resume', '/work/instance-nvme/runs/'+parent+'/checkpoint.pt',
                 '--steps', str(6496-1152), '--decay-start', str(count(5846)), '--decay-end', str(count(6496)),
                 '--extend-stable-schedule', '--checkpoint-every', '1024']
        name = f'{identity}-{arm}-s419-u6496'
        module = 'scripts.train_reinjection_control' if arm=='reinjection' else 'scripts.train_primitive_fp32'
        row = native(name, module, args, half)
        if not row['all_replica_model_optimizer_states_equal'] or row['total_real_targets']!=count(6496):
            raise AssertionError('Resource control state/exposure differs')
        _, result = evaluate(name, arm, half, 'curve')
        write_state(public/(arm+'-matched-native.json'), result)
    def fixed_arguments(lr, seed, tune=False):
        return ['--corpus', '/work/instance-nvme/natural-append-6500m-v2/source.json',
            '--packed-corpus', '/work/instance-nvme/natural-append-6500m-v2/packed',
            '--dev-corpus', '/work/instance-nvme/variable-depth-fresh-v1/dev.json',
            '--architecture', 'preln', '--repetitions', '2', '--depth-policy', 'fixed', '--precision', 'fp32',
            '--context', '1024', '--batch', '1', '--accumulation', '8', '--optimizer', 'fused', '--recipe', 'pretraining',
            '--seed', str(seed), '--data-seed', '23', '--lr', str(lr), '--lr-warmup-tokens', str(int(count(1152 if tune else 3840)*.025)),
            '--decay-start', str(count(1037 if tune else 3456)), '--decay-end', str(count(1152 if tune else 3840)),
            '--steps', '1152' if tune else '3840', '--warmup', '4', '--reference-threads', '8',
            '--purpose', 'tuning' if tune else 'scientific', '--save', '--checkpoint-every', '1024']
    def tune_fixed():
        rows = []
        for i, lr in enumerate(plan['lr_grid']):
            name = f'{identity}-fixed-tune-lr{i}-s11'
            row = native(name, 'scripts.train_primitive_fp32', fixed_arguments(lr, 11, True), 1)
            if row['initial_model_digest'] != plan['initial_tune_digest'] or row['total_real_targets']!=count(1152):
                raise AssertionError('Fixed TUNE initialization/exposure mismatch')
            evaluation, result = evaluate(name, 'fixed', 1, 'tune')
            rows.append({'identity': name, 'evaluation_identity': evaluation, 'arm': 'fixed', 'seed': 11,
                         'endpoint': 'tune301M', 'lr': lr, 'metrics': result['metrics']})
            write_state(public/'tuning.json', {'rows': rows})
        best = min(r['metrics']['weighted_nll'] for r in rows)
        return min((r for r in rows if r['metrics']['weighted_nll']<=best+.0002), key=lambda r:r['lr'])
    def fixed_main(selected):
        name = f'{identity}-fixed-s419-u3840'
        row = native(name, 'scripts.train_primitive_fp32', fixed_arguments(selected['lr'], 419), 1)
        if (row['initial_model_digest']!=plan['initial_main_digest'] or row['total_real_targets']!=count(3840)
                or not row['all_replica_model_optimizer_states_equal']):
            raise AssertionError('Fixed main initialization/state/exposure differs')
        _, result = evaluate(name, 'fixed', 1, 'curve')
        write_state(public/'fixed-native.json', result)
    try:
        shadow = audit(models, 'main-independent-cpu', 'curve')
        with cf.ThreadPoolExecutor(2) as pool:
            base = pool.submit(continuation, 'reinjection', 0)
            tune = pool.submit(tune_fixed)
            selected = tune.result()
            base.result()
        write_state(public/'selected.json', selected)
        selected_audit = audit([selected], 'tune-independent-cpu', 'tune')
        selected_audit.wait(timeout=max(1, min(1800, deadline-time.time()-60)))
        if selected_audit.returncode: raise RuntimeError('Selected fixed TUNE audit failed')
        write_state(public/'tune-independent-cpu.json', json.loads((out/'tune-independent-cpu/report.json').read_text()))
        with cf.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(continuation, 'ordinary', 0), pool.submit(fixed_main, selected)]
            for future in cf.as_completed(futures): future.result()
        shadow.wait(timeout=max(1, deadline-time.time()-60))
        if shadow.returncode: raise RuntimeError('Canonical endpoint audit failed')
        frozen()
        write_state(public/'independent-cpu-verification.json', json.loads((out/'main-independent-cpu/report.json').read_text()))
        report = {'status': 'COMPLETE_COMMON_FP32_DEPTH_CONNECTION_COMPARISON', 'test_accessed': False,
                  'new_terminal_attempt': False, 'scope': plan['scope']}
    except BaseException as error:
        report = {'status': 'STOPPED_COMMON_FP32_DEPTH_CONNECTION_COMPARISON', 'error': str(error), 'test_accessed': False}
        raise
    finally:
        for name, container in list(active.items()):
            try: cancel_worker_container(container, name, 'campaign_stop')
            except BaseException: pass
        for proc in cpu:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
        report['wall_seconds'] = time.time()-started
        write_state(out/'report.json', report)
        write_state(public/'report.json', report)
        write_state(public/'execution-costs.json', {'status': report['status'], 'attempts': costs,
            'wall_seconds': time.time()-started, 'scope': 'All tuning, retained/discarded native paths and failures included. CPU audits/environment/isolation have separate records. Overlapping intervals are not summed as sequential node time.'})


if __name__ == '__main__':
    main()
