"""Conditional paired FP16-body gradient check; never starts research training."""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import torch

from compute_trainium.execution_isolation import cancel_worker_container, managed_running
from compute_trainium.owned_native_phase import create_worker
from compute_trainium.parcae_update_qualification import relative_error
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.screen_parcae_fp16_cpu import group


def launch_ready(gradient, optimizer, managed_active, now, latest):
    if now > latest:
        raise TimeoutError('FP16 latest-start window expired')
    for record, expected in [(gradient, 'PASS_PAIRED_NATIVE_PARCAE_STATIC_STAGE_GLOBAL_GRADIENTS'),
                             (optimizer, 'PASS_PAIRED_NATIVE_PARCAE_OPTIMIZER_AND_RECOVERY')]:
        if record and record['status'] != expected:
            raise RuntimeError('Prior native numerical gate failed; no FP16 attempt starts')
    return bool(gradient) and bool(optimizer) and not managed_active


def compare(reference, candidate):
    expected = torch.load(reference, map_location='cpu', mmap=True, weights_only=False)
    actual = torch.load(candidate, map_location='cpu', mmap=True, weights_only=False)
    delta = relative_error(expected['gradients'], actual['gradients'])
    families = {}
    for category in sorted({group(n) for n in expected['gradients']}):
        names = [n for n in expected['gradients'] if group(n) == category]
        families[category] = relative_error({n: expected['gradients'][n] for n in names},
                                          {n: actual['gradients'][n] for n in names})
    nll = actual['loss']-expected['loss']
    passed = delta['relative_l2'] <= .02 and abs(nll) <= .003 and all(r['relative_l2'] <= .02 for r in families.values())
    return {'global_gradient': delta, 'gradient_groups': families,
        'native_minus_cpu_nll': nll, 'passes_registered_gates': passed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    reference = args.root / 'parcae-native-global-v1'
    optimizer = args.root / 'parcae-optimizer-global-v1/attempt2'
    cpu_root = args.root / 'parcae-cpu-fp16-v1'
    out = args.root / 'parcae-native-fp16-v1'
    public = Path('results/parcae_native_fp16_20261024')
    out.mkdir(exist_ok=True); public.mkdir(parents=True, exist_ok=True)
    cpu = json.loads((cpu_root / 'report.json').read_text())
    if cpu['status'] != 'PASS_CPU_PARCAE_FP16_BODY_SCREEN':
        raise AssertionError('CPU FP16-body screen failed')
    plan_file = out / 'plan.json'
    if not plan_file.exists():
        hashes = {}
        for file in [cpu_root / 'plan.json', reference / 'native-execution-plan.json', optimizer / 'native-plan.json']:
            for path, sha in json.loads(file.read_text())['source_sha256'].items():
                if path in hashes and hashes[path] != sha:
                    raise AssertionError('Prerequisite source freezes disagree')
                hashes[path] = sha
        files = ['scripts/run_parcae_fp16_native.py', 'scripts/capture_parcae_fp16_global.py',
                 'docs/PARCAE_FP16_QUALIFICATION_20261024.md']
        hashes.update({f: file_sha256(f) for f in files})
        plan = {'identity': 'parcae-native-fp16-v1', 'registered_unix': time.time(),
            'source_sha256': hashes, 'cpu_screen_report_sha256': file_sha256(cpu_root / 'report.json'),
            'cpu_screen_plan_sha256': file_sha256(cpu_root / 'plan.json'),
            'local_reference_sha256': {arm: file_sha256(reference / arm / 'local-cpu-fp32.pt')
                                       for arm in ['ordinary', 'recurrent']},
            'reference_plan_sha256': file_sha256(reference / 'plan.json'),
            'reference_report_sha256': file_sha256(reference / 'report.json'),
            'latest_start_unix': dt.datetime(2026, 10, 25, 3, tzinfo=dt.timezone.utc).timestamp(),
            'native_seconds_limit': 1800, 'gradient_relative_gate': .02, 'nll_absolute_gate': .003,
            'manual_loss_scale': 8192, 'optimizer_updates': 0, 'test_accessed': False,
            'scope': 'Paired all16-chip FP16-autocast body/FP32-head gradient qualification after native FP32 gradient and optimizer/recovery pass. No research training, throughput claim or precision adoption.'}
        write_state(plan_file, plan); write_state(public / 'plan.json', plan)
    plan = json.loads(plan_file.read_text())

    def frozen():
        if any(file_sha256(f) != sha for f,sha in plan['source_sha256'].items()):
            raise AssertionError('FP16 qualification source changed')
        for file, key in [(cpu_root / 'report.json', 'cpu_screen_report_sha256'),
                          (cpu_root / 'plan.json', 'cpu_screen_plan_sha256'),
                          (reference / 'plan.json', 'reference_plan_sha256'),
                          (reference / 'report.json', 'reference_report_sha256')]:
            if file_sha256(file) != plan[key]:
                raise AssertionError('FP16 CPU/reference preparation changed')
    frozen()
    if args.prepare_only:
        return
    if (out / 'report.json').exists():
        raise FileExistsError('No silent FP16 native rerun')
    waiting = time.time()
    started = None
    active, records, costs = {}, [], []
    lock = threading.Lock()
    failed = threading.Event()

    def capture(arm, half, deadline):
        try:
            frozen()
            if time.time() > plan['latest_start_unix']:
                raise TimeoutError('FP16 phase latest-start window expired')
            identity = 'parcae-native-fp16-v1-' + arm
            container = create_worker(args.root, identity, half)
            with lock:
                active[identity] = container
            proc = None
            try:
                env = dict(os.environ, RESEARCH_ROOT=str(args.root), RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                    RESEARCH_RUN=identity, RESEARCH_OWNER=identity, RESEARCH_CONTAINER=container,
                    RESEARCH_ENTRY_MODULE='scripts.capture_parcae_fp16_global', RESEARCH_DDP_RANKS='32',
                    NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,half*32+32))),
                    RESEARCH_TIMEOUT_SECONDS=str(max(1,int(deadline-time.time()))))
                with (out / (arm+'.log')).open('x') as log:
                    if time.time() > plan['latest_start_unix']:
                        raise TimeoutError('FP16 container setup crossed latest start')
                    proc = subprocess.Popen([os.sys.executable, '-u', '-m', 'scripts.supervise_isolated',
                        '--reference-root', '/work/instance-nvme/parcae-native-global-v1',
                        '--corpus', '/work/instance-nvme/natural-append-6500m-v2/source.json',
                        '--packed-corpus', '/work/instance-nvme/natural-append-6500m-v2/packed', '--arm', arm],
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    if failed.is_set() or time.time() > deadline:
                        raise RuntimeError('FP16 paired stop or thirty-minute deadline')
                    time.sleep(5)
                if proc.returncode:
                    raise RuntimeError('Native FP16 capture failed: '+arm)
                result = json.loads((args.root / 'runs' / identity / 'report.json').read_text())
                if not result['weights_unchanged'] or not result['all_replica_gradients_equal'] or result['manual_loss_scale'] != 8192:
                    raise AssertionError('FP16 capture identity/replica/scale gate failed')
                reference_proof = json.loads((reference / 'report.json').read_text())
                ref = next(r for r in reference_proof['records'] if r['arm'] == arm)
                if file_sha256(reference / arm / 'global-cpu-fp32.pt') != ref['global_gradient_file_sha256'] or result['real_targets'] != ref['real_targets']:
                    raise AssertionError('FP16 independent gradient/exposure changed')
                if file_sha256(reference / arm / 'local-cpu-fp32.pt') != plan['local_reference_sha256'][arm]:
                    raise AssertionError('Independent local FP32 reference changed')
                row = {'arm': arm}
                for kind in ['local', 'global']:
                    row[kind] = compare(reference / arm / (kind+'-cpu-fp32.pt'),
                        args.root / 'runs' / identity / (kind+'-native.pt'))
                row['passes_registered_gates'] = row['local']['passes_registered_gates'] and row['global']['passes_registered_gates']
                with lock:
                    records.append(row); write_state(public / 'progress.json', {'records': records})
                if not row['passes_registered_gates']:
                    raise AssertionError('Native FP16 numerical gradient gate failed: '+arm)
            finally:
                cancel_worker_container(container, identity, 'fp16_capture_complete')
                if proc is not None and proc.poll() is None:
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                closure = args.root / 'snapshots' / identity / 'closure.json'
                with lock:
                    active.pop(identity, None)
                    costs.append({'identity': identity, **json.loads(closure.read_text())} if closure.exists() else {'identity': identity, 'closure_missing': True})
        except BaseException:
            failed.set()
            raise

    try:
        while True:
            frozen()
            gfile, ofile = reference / 'native-report.json', optimizer / 'native-report.json'
            gradient = json.loads(gfile.read_text()) if gfile.exists() else None
            opt = json.loads(ofile.read_text()) if ofile.exists() else None
            if launch_ready(gradient, opt, bool(managed_running()) if gradient and opt else True,
                            time.time(), plan['latest_start_unix']):
                break
            write_state(public / 'status.json', {'status': 'WAITING_FOR_FP32_GRADIENT_AND_OPTIMIZER_GATES',
                'observed_unix': time.time(), 'assigned_chips': 0})
            time.sleep(30)
        started = time.time()
        write_state(public / 'status.json', {'status': 'RUNNING_NATIVE_FP16_BODY_GRADIENT_CAPTURE',
            'started_unix': started, 'maximum_chips': 16})
        deadline = started+plan['native_seconds_limit']
        with cf.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(capture, arm, i, deadline) for i, arm in enumerate(['ordinary', 'recurrent'])]
            for future in cf.as_completed(futures):
                future.result()
        frozen()
        if time.time() > deadline:
            raise TimeoutError('FP16 final comparisons exceeded native window')
        result = {'status': 'PASS_PAIRED_NATIVE_PARCAE_FP16_BODY_GRADIENTS', 'records': records}
    except BaseException as error:
        result = {'status': 'STOPPED_NATIVE_PARCAE_FP16_BODY_QUALIFICATION', 'error': str(error), 'records': records}
        raise
    finally:
        for identity, container in list(active.items()):
            try:
                cancel_worker_container(container, identity, 'fp16_qualification_stop')
            except BaseException:
                pass
        now = time.time()
        result.update(waiting_seconds=(started or now)-waiting,
            active_execution_seconds=now-started if started else 0., optimizer_updates=0,
            test_accessed=False, scope=plan['scope'])
        write_state(out / 'report.json', result); write_state(public / 'report.json', result)
        write_state(public / 'status.json', {'status': result['status'], 'observed_unix': now, 'assigned_chips': 0})
        write_state(public / 'execution-costs.json', {'status': result['status'], 'attempts': costs,
            'waiting_seconds': result['waiting_seconds'], 'active_execution_seconds': result['active_execution_seconds'],
            'scope': 'Waiting assigns zero chips. Native receipts include compilation/capture and failures. CPU screen separate; do not add overlapping wall intervals.'})


if __name__ == '__main__':
    main()
