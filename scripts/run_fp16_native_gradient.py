"""Ownership-isolated paired FP16 full-gradient qualification, no updates."""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from compute_trainium.execution_isolation import metadata, verify_owner, cancel_worker_container
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.capture_gradient_precision_control import compare_gradients


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    identity = 'fp16-native-gradient-v1'
    out = a.root/identity
    out.mkdir(exist_ok=True)
    public = Path('results/fp16_native_gradient_20261024')
    public.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    started = time.time()
    if json.loads(Path('results/native_execution_isolation_20261024/report.json').read_text())['status'] != 'PASS_SEPARATE_CONTAINER_SCOPED_CANCELLATION':
        raise ValueError('Actual scoped-cancellation qualification missing')
    if not (out/'plan.json').exists():
        rows = []
        for half, arm in enumerate(['ordinary', 'fixed']):
            old = f'trained-gradient-v1-{arm}-reference'
            ref = json.loads((a.root/'runs'/old/'report.json').read_text())
            args = json.loads((a.root/'snapshots'/old/'launch.json').read_text())['arguments']
            source = args[args.index('--checkpoint')+1]
            host = a.root/source.removeprefix('/work/instance-nvme/')
            if file_sha256(host) != ref['checkpoint_sha256']: raise AssertionError('Mature source changed')
            rows.append({'arm': arm, 'half': half, 'identity': identity+'-'+arm,
                'container': 'ac-native-fp16-global-v1-'+arm, 'source_checkpoint': source,
                'checkpoint_sha256': ref['checkpoint_sha256'], 'source_update': ref['source_update'],
                'reference_identity': old, 'reference_gradient_sha256': file_sha256(a.root/'runs'/old/'global-cpu-fp32.pt'),
                'reference_report_sha256': file_sha256(a.root/'runs'/old/'report.json'),
                'real_targets': ref['real_targets']})
        files = ['scripts/capture_fp16_head_gradient.py', 'scripts/run_fp16_native_gradient.py',
                 'scripts/run_native_isolated.sh', 'scripts/supervise_isolated.py',
                 'compute_trainium/execution_isolation.py', 'scripts/capture_gradient_precision_control.py',
                 'docs/FP16_NATIVE_GRADIENT_PROTOCOL_20261024.md']
        hashes = dict(json.loads(Path('results/mature_fp32_20261024/plan.json').read_text())['source_sha256'])
        hashes.update({f: file_sha256(f) for f in files})
        plan = {'identity': identity, 'registered_unix': time.time(), 'rows': rows,
                'source_sha256': hashes, 'maximum_native_and_reference_seconds': 1800,
                'gradient_relative_l2_limit': .02, 'nll_absolute_limit': .003,
                'optimizer_updates': 0, 'test_accessed': False,
                'scope': 'Full256-window native FP16-body/FP32-head scaled-gradient capture against pinned CPU FP32, separately isolated worker containers. No optimizer, recovery, throughput or learning qualification.'}
        write_state(out/'plan.json', plan)
        write_state(public/'plan.json', plan)
    plan = json.loads((out/'plan.json').read_text())
    def frozen():
        if any(file_sha256(f) != h for f, h in plan['source_sha256'].items()):
            raise AssertionError('Registered numerical source changed')
    frozen()
    for row in plan['rows']:
        verify_owner(metadata(row['container']), row['identity'])
    if a.prepare_only: return
    if (out/'report.json').exists(): raise FileExistsError('No silent native qualification rerun')
    for arm in ['ordinary', 'reinjection']:
        path = a.root/'snapshots'/f'reinjection-pilot-v1-{arm}-s419-u3840-curve-r1'/'closure.json'
        if not path.exists() or json.loads(path.read_text())['exit_code']:
            raise ValueError('Previous final native science owner has not closed cleanly')
    deadline = time.time()+plan['maximum_native_and_reference_seconds']
    attempts = []
    records = []
    report = None
    def capture(row):
        frozen()
        name, half = row['identity'], row['half']
        if (a.root/'runs'/name).exists(): raise FileExistsError(name)
        env = dict(os.environ, RESEARCH_ROOT=str(a.root), RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
            RESEARCH_RUN=name, RESEARCH_ENTRY_MODULE='scripts.capture_fp16_head_gradient',
            RESEARCH_DDP_RANKS='32', NEURON_RT_VISIBLE_CORES=','.join(map(str, range(half*32, half*32+32))),
            RESEARCH_CONTAINER=row['container'], RESEARCH_OWNER=name,
            RESEARCH_TIMEOUT_SECONDS=str(int(deadline-time.time())))
        args = ['--checkpoint', row['source_checkpoint'], '--expected-checkpoint-sha256', row['checkpoint_sha256'],
                '--corpus', '/work/instance-nvme/natural-append-6500m-v2/source.json',
                '--packed-corpus', '/work/instance-nvme/natural-append-6500m-v2/packed']
        with (out/(name+'.log')).open('x') as log:
            proc = subprocess.Popen([os.sys.executable, '-u', '-m', 'scripts.supervise_isolated', *args],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        write_state(out/(name+'-process.json'), {'pid': proc.pid, 'started_unix': time.time()})
        try:
            code = proc.wait(timeout=max(1, deadline-time.time()+35))
        except BaseException:
            cancel_worker_container(row['container'], name, 'capture_failure')
            proc.wait(timeout=30)
            raise
        closure = json.loads((a.root/'snapshots'/name/'closure.json').read_text())
        return {'identity': name, **closure}, code
    try:
        with cf.ThreadPoolExecutor(2) as pool:
            tasks = {pool.submit(capture, row): row for row in plan['rows']}
            errors = []
            for task in cf.as_completed(tasks):
                row = tasks[task]
                try:
                    receipt, code = task.result()
                    attempts.append(receipt)
                    if code: raise RuntimeError('Native capture failed for '+row['arm'])
                    native_report = json.loads((a.root/'runs'/row['identity']/'report.json').read_text())
                    refdir = a.root/'runs'/row['reference_identity']
                    if (file_sha256(refdir/'global-cpu-fp32.pt') != row['reference_gradient_sha256']
                            or file_sha256(refdir/'report.json') != row['reference_report_sha256']):
                        raise AssertionError('Pinned complete CPU reference changed')
                    native = torch.load(a.root/'runs'/row['identity']/'global-native.pt', map_location='cpu', mmap=True, weights_only=False)
                    reference = torch.load(refdir/'global-cpu-fp32.pt', map_location='cpu', mmap=True, weights_only=False)
                    if native_report['real_targets'] != row['real_targets'] or native_report['source_update'] != row['source_update']:
                        raise AssertionError('Global real-token/source counter mismatch')
                    comparison = compare_gradients(native['gradients'], reference['gradients'])
                    delta = native['loss']-reference['loss']
                    passed = comparison['relative_l2'] <= .02 and abs(delta) <= .003
                    records.append({'arm': row['arm'], 'source_update': row['source_update'],
                        'checkpoint_sha256': row['checkpoint_sha256'], 'real_targets': row['real_targets'],
                        'gradient_comparison': comparison, 'native_minus_cpu_nll': delta,
                        'passes_registered_global_gradient_and_nll_gate': passed,
                        'all_replica_gradients_equal': native_report['all_replica_gradients_equal'],
                        'weights_unchanged': native_report['weights_unchanged']})
                    del native, reference
                except BaseException as error:
                    errors.append({'arm': row['arm'], 'error': str(error)})
                write_state(public/'progress.json', {'records': records, 'errors': errors, 'attempts': attempts})
        frozen()
        report = {'status': 'COMPLETE_NATIVE_FP16_GLOBAL_GRADIENT_SCREEN' if not errors else 'STOPPED_NATIVE_FP16_GLOBAL_GRADIENT_SCREEN',
                  'records': records, 'errors': errors,
                  'qualifies_both_mature_global_objectives': len(records)==2 and all(r['passes_registered_global_gradient_and_nll_gate'] for r in records),
                  'optimizer_updates': 0, 'test_accessed': False, 'scope': plan['scope']}
    finally:
        cleanup_errors = []
        for row in plan['rows']:
            try:
                cancel_worker_container(row['container'], row['identity'], 'execution_complete')
            except BaseException as error:
                cleanup_errors.append({'arm': row['arm'], 'error': str(error)})
            closure = a.root/'snapshots'/row['identity']/'closure.json'
            if closure.exists() and not any(r['identity']==row['identity'] for r in attempts):
                attempts.append({'identity': row['identity'], **json.loads(closure.read_text())})
        if report is None: report = {'status': 'STOPPED_NATIVE_FP16_CONTROLLER_ERROR', 'optimizer_updates': 0, 'test_accessed': False}
        if cleanup_errors:
            report['status'] = 'STOPPED_NATIVE_FP16_CLEANUP_ERROR'
            report['cleanup_errors'] = cleanup_errors
        report['wall_seconds'] = time.time()-started
        write_state(out/'report.json', report)
        write_state(public/'report.json', report)
        write_state(public/'execution-costs.json', {'status': report['status'], 'attempts': attempts,
            'wall_seconds': time.time()-started, 'scope': 'Paired isolated captures; no optimizer/model updates. Environment copy and CPU cancellation fixture have separate records.'})


if __name__ == '__main__':
    main()
