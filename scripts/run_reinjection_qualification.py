"""Bounded actual-size qualification of the inexpensive trained control."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    identity = 'reinjection-native-q-v1'
    output = a.root/identity
    output.mkdir(exist_ok=True)
    public = Path('results/reinjection_control_20261024/native-v1')
    public.mkdir(exist_ok=True)
    started = time.time()
    fixture = json.loads(Path('results/reinjection_control_20261024/cpu-fixture-v2-report.json').read_text())
    if fixture['status'] != 'PASS_INDEPENDENT_CPU_REINJECTION_FACTORY_FIXTURE':
        raise ValueError('Independent CPU factory agreement missing')
    files = ['compute_trainium/reinjection_control.py', 'scripts/train_reinjection_control.py',
             'scripts/evaluate_reinjection_control.py', 'scripts/score_reinjection_control.py',
             'scripts/run_reinjection_qualification.py', 'docs/TRAINED_REINJECTION_CONTROL_20261024.md']
    if not (output/'plan.json').exists():
        deps = json.loads(Path('results/mature_fp32_20261024/plan.json').read_text())['source_sha256']
        hashes = {**deps, **{f: file_sha256(f) for f in files}}
        plan = {'identity': identity, 'registered_unix': time.time(), 'source_sha256': hashes,
                'corpus_manifest_sha256': file_sha256(a.root/'natural-append-6500m-v2/manifest.json'),
                'cpu_fixture_sha256': file_sha256('results/reinjection_control_20261024/cpu-fixture-v2-report.json'),
                'phases': ['global32-eight-updates', 'perf256-104-updates', 'restore256-eight-updates'],
                'maximum_native_sequence_seconds': 1500, 'assigned_chips': 8,
                'scope': 'Qualification only, no scientific exposure or held-out scoring. Separate actual32-window global check and256-window throughput.'}
        write_state(output/'plan.json', plan)
        write_state(public/'plan.json', plan)
    plan = json.loads((output/'plan.json').read_text())
    def frozen():
        if any(file_sha256(f) != h for f, h in plan['source_sha256'].items()):
            raise AssertionError('Qualification source changed')
        if file_sha256(a.root/'natural-append-6500m-v2/manifest.json') != plan['corpus_manifest_sha256']:
            raise AssertionError('Source changed')
    frozen()
    if a.prepare_only: return
    if (output/'report.json').exists(): raise FileExistsError('No silent qualification rerun')
    attempts = []
    report = None
    proc = None
    try:
        prior = a.root/'snapshots/mature-fp32-v1-fixed-s103-u11520-dev-r2/closure.json'
        while not prior.exists():
            failed = a.root/'mature-fp32-v1/report.json'
            if failed.exists() and json.loads(failed.read_text())['status'].startswith('STOPPED'):
                raise RuntimeError('Previous fixed native owner failed')
            if time.time()-started > 3600: raise TimeoutError('Previous core owner did not close')
            write_state(output/'heartbeat.json', {'phase': 'waiting_for_previous_fixed_native_owner', 'unix': time.time()})
            time.sleep(10)
        if json.loads(prior.read_text())['exit_code']:
            raise RuntimeError('Previous fixed native readout failed')
        native_started = time.time()
        deadline = native_started+plan['maximum_native_sequence_seconds']
        base = ['--corpus', '/work/instance-nvme/natural-append-6500m-v2/source.json',
                '--packed-corpus', '/work/instance-nvme/natural-append-6500m-v2/packed',
                '--architecture', 'preln', '--depth-policy', 'ordinary', '--repetitions', '1',
                '--precision', 'fp32', '--context', '1024', '--batch', '1', '--optimizer', 'fused',
                '--recipe', 'pretraining', '--seed', '739', '--data-seed', '23', '--lr', '.0006',
                '--lr-warmup-tokens', '10000000', '--purpose', 'qualification', '--warmup', '4',
                '--reference-threads', '8', '--save']
        rows = []
        for phase, steps, accumulation in [('global32', 8, 1), ('perf256', 104, 8), ('restore256', 8, 8)]:
            frozen()
            name = identity+'-'+phase
            if (a.root/'runs'/name).exists(): raise FileExistsError(name)
            args = base+['--steps', str(steps), '--accumulation', str(accumulation)]
            if phase == 'global32': args += ['--check-global']
            if phase == 'restore256':
                args += ['--resume', '/work/instance-nvme/runs/'+identity+'-perf256/checkpoint.pt']
            env = dict(os.environ, RESEARCH_ROOT=str(a.root), RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                RESEARCH_RUN=name, RESEARCH_ENTRY_MODULE='scripts.train_reinjection_control',
                RESEARCH_DDP_RANKS='32', NEURON_RT_VISIBLE_CORES=','.join(map(str, range(32, 64))),
                RESEARCH_TIMEOUT_SECONDS=str(int(deadline-time.time())))
            if time.time()+120 > deadline: raise TimeoutError('Full qualification window exhausted')
            with (output/(name+'.log')).open('x') as log:
                proc = subprocess.Popen([os.sys.executable, '-u', '-m', 'scripts.supervise', *args],
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
            write_state(output/(name+'-process.json'), {'pid': proc.pid, 'started_unix': time.time()})
            code = proc.wait(timeout=max(1, deadline-time.time()+30))
            closure = json.loads((a.root/'snapshots'/name/'closure.json').read_text())
            attempts.append({'identity': name, **closure})
            write_state(public/'execution-costs.json', {'status': 'RUNNING', 'attempts': attempts})
            if code: raise RuntimeError('Native qualification failed: '+phase)
            proc = None
            row = json.loads((a.root/'runs'/name/'report.json').read_text())
            if not row['all_replica_model_optimizer_states_equal'] or row['parameters'] != 150436608:
                raise AssertionError('State/parameter gate failed')
            if not row['signature']['model'].get('reinjection_only'):
                raise AssertionError('Distinct model marker missing')
            if phase == 'global32' and row['global_gradient_check']['gradient_relative_l2'] > .02:
                raise AssertionError('Global gradient gate failed')
            if phase == 'restore256' and row['recovered_model_optimizer_digest'] != rows[-1]['state_digest']:
                raise AssertionError('Complete model/optimizer load differs')
            rows.append(row)
            write_state(public/(phase+'-report.json'), row)
        frozen()
        report = {'status': 'PASS_REINJECTION_NATIVE_QUALIFICATION', 'rows': rows,
                  'measured_warm_targets_per_second': rows[1]['warm_global_real_targets_per_second'],
                  'native_sequence_seconds': time.time()-native_started,
                  'wall_seconds_including_core_wait': time.time()-started,
                  'test_accessed': False, 'scope': plan['scope']}
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            subprocess.run(['docker', 'stop', '--time', '5', 'autonomous-compute-neuron233'],
                           capture_output=True, timeout=25)
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: proc.terminate()
        if 'name' in locals() and not any(r['identity'] == name for r in attempts):
            closure = a.root/'snapshots'/name/'closure.json'
            if closure.exists(): attempts.append({'identity': name, **json.loads(closure.read_text())})
        report = {'status': 'STOPPED_REINJECTION_NATIVE_QUALIFICATION', 'error': str(error),
                  'wall_seconds_including_core_wait': time.time()-started, 'test_accessed': False}
        raise
    finally:
        write_state(output/'report.json', report)
        write_state(public/'report.json', report)
        write_state(public/'execution-costs.json', {'status': report['status'], 'attempts': attempts,
                    'wall_seconds_including_core_wait': time.time()-started,
                    'scope': 'Eight-chip qualification assignments overlap other fits; no qualification update counts as scientific exposure.'})


if __name__ == '__main__':
    main()
