"""Missing ordinary resource control, on a separately registered half-node."""
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
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
    identity = 'mature-fp32-compute-v1'
    output = a.root/identity
    output.mkdir(exist_ok=True)
    public = Path('results/mature_fp32_compute_20261024')
    public.mkdir(exist_ok=True)
    started = time.time()
    prior_plan = json.loads((a.root/'mature-fp32-v1/plan.json').read_text())
    source = a.root/'runs/mature-fp32-v1-ordinary-s103-u10368/checkpoint.pt'
    audit_rows = json.loads((a.root/'mature-fp32-v1/independent-cpu/progress.json').read_text())['records']
    anchor = next(r for r in audit_rows if r['identity'] == source.parent.name)
    if not anchor['copy_checksum_and_full_state_readback'] or file_sha256(source) != anchor['checkpoint_sha256']:
        raise AssertionError('Stable parent has not passed its independent state audit')
    prefix = np.load(a.root/'natural-append-6500m-v2/packed/prefix_counts.npy', mmap_mode='r')
    def count(u): return int(prefix[u*256].sum())
    cutoff = dt.datetime(2026, 10, 24, 14, 45, tzinfo=dt.timezone.utc).timestamp()
    deadline = dt.datetime(2026, 10, 24, 15, 30, tzinfo=dt.timezone.utc).timestamp()
    rate = prior_plan['budget']['measured_paired_targets_per_second']['ordinary']
    budget = ((count(19490)-count(10368))/rate+2*195+2*100)*1.3+1800
    if not (output/'plan.json').exists():
        hashes = dict(prior_plan['source_sha256'])
        for f in [__file__, 'docs/MATURE_FP32_COMPUTE_CONTROL_20261024.md']:
            path = str(Path(f).relative_to(Path.cwd())) if Path(f).is_absolute() else f
            hashes[path] = file_sha256(f)
        plan = {'identity': identity, 'registered_unix': time.time(),
                'source_sha256': hashes, 'parent_sha256': anchor['checkpoint_sha256'],
                'dev_sha256': prior_plan['dev_sha256'],
                'corpus_manifest_sha256': prior_plan['corpus_manifest_sha256'],
                'updates': [10368, 17541, 19490],
                'targets': {str(u): count(u) for u in [10368, 17541, 19490]},
                'parameters': 150436608, 'ordinary_step_flops': prior_plan['step_flops']['ordinary'],
                'fixed3B_matmul_flops': 11520*prior_plan['step_flops']['fixed'],
                'ordinary_matmul_flops': 19490*prior_plan['step_flops']['ordinary'],
                'measured_targets_per_second': rate, 'budget_seconds': budget,
                'training_cutoff_unix': cutoff, 'hard_deadline_unix': deadline,
                'scope': 'One paired reused seed, consumed DEV. Properly annealed ordinary resource control; no TEST or terminal confirmation.'}
        write_state(output/'plan.json', plan)
        write_state(public/'plan.json', plan)
    else:
        plan = json.loads((output/'plan.json').read_text())
    def frozen():
        if any(file_sha256(f) != h for f, h in plan['source_sha256'].items()):
            raise AssertionError('Saved source changed')
        if file_sha256(source) != plan['parent_sha256']:
            raise AssertionError('Stable parent changed')
        if (file_sha256(a.root/'variable-depth-fresh-v1/dev.json') != plan['dev_sha256']
                or file_sha256(a.root/'natural-append-6500m-v2/manifest.json') != plan['corpus_manifest_sha256']):
            raise AssertionError('Data/panel changed')
    frozen()
    if time.time()+plan['budget_seconds'] > deadline:
        raise TimeoutError('Full measured resource-control budget no longer fits')
    if a.prepare_only:
        write_state(output/'validation.json', {'status': 'PASS_PREPARED_RESOURCE_CONTROL_NO_UPDATES', 'unix': time.time()})
        return
    if (output/'report.json').exists():
        raise FileExistsError('Closed identity; no silent rerun')
    inventory = [{'identity': f'{identity}-ordinary-s103-u{u}', 'arm': 'ordinary',
                  'seed': 103, 'endpoint': 'matched3B' if u == 19490 else 'stable',
                  'parameters': 150436608} for u in [17541, 19490]]
    write_state(output/'models.json', inventory)
    write_state(public/'models.json', inventory)
    active = audit = None
    costs = []
    report = None
    try:
        old = 'mature-fp32-v1-ordinary-s103-u11520-dev-r1'
        while not (a.root/'snapshots'/old/'closure.json').exists():
            closed = a.root/'mature-fp32-v1/report.json'
            if closed.exists() and json.loads(closed.read_text())['status'].startswith('STOPPED'):
                raise RuntimeError('Prior paired diagnostic stopped')
            if time.time()+plan['budget_seconds'] > deadline:
                raise TimeoutError('Waiting for prior core owner exhausted full budget')
            write_state(output/'heartbeat.json', {'unix': time.time(), 'phase': 'waiting_for_previous_ordinary_native_owner'})
            time.sleep(10)
        if json.loads((a.root/'snapshots'/old/'closure.json').read_text())['exit_code']:
            raise RuntimeError('Prior ordinary native readout failed')
        with (output/'independent-cpu.log').open('x') as log:
            audit = subprocess.Popen([os.sys.executable, '-u', '-m', 'scripts.audit_variable_depth',
                '--root', str(a.root), '--models', str(output/'models.json'),
                '--corpus', str(a.root/'variable-depth-fresh-v1/dev.json'),
                '--output', str(output/'independent-cpu'), '--deadline-unix', str(deadline-60)],
                env=dict(os.environ, USE_TORCH_XLA='0', OMP_NUM_THREADS='12', MKL_NUM_THREADS='12'),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_state(output/'independent-cpu-process.json', {'pid': audit.pid, 'started_unix': time.time()})
        def native(name, module, args, evaluation=False):
            nonlocal active
            frozen()
            if (a.root/'runs'/name).exists(): raise FileExistsError(name)
            limit = deadline if evaluation else cutoff
            cores = list(range(0, 32, 4 if evaluation else 1))
            env = dict(os.environ, RESEARCH_ROOT=str(a.root), RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                RESEARCH_RUN=name, RESEARCH_ENTRY_MODULE=module, RESEARCH_DDP_RANKS=str(len(cores)),
                NEURON_RT_VISIBLE_CORES=','.join(map(str, cores)),
                RESEARCH_TIMEOUT_SECONDS=str(min(15000, int(limit-time.time()))))
            with (output/(name+'.log')).open('x') as log:
                active = subprocess.Popen([os.sys.executable, '-u', '-m', 'scripts.supervise', *args],
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
            write_state(output/(name+'-process.json'), {'pid': active.pid, 'started_unix': time.time()})
            while active.poll() is None:
                if time.time() > limit or (audit.poll() is not None and audit.returncode):
                    raise TimeoutError('Native deadline or independent audit failure')
                write_state(output/'heartbeat.json', {'unix': time.time(), 'phase': name})
                time.sleep(10)
            closure = json.loads((a.root/'snapshots'/name/'closure.json').read_text())
            costs.append({'identity': name, **closure})
            write_state(public/'execution-costs.json', {'status': 'RUNNING', 'attempts': costs})
            if active.returncode: raise RuntimeError('Native stage failed: '+name)
            active = None
            return json.loads((a.root/'runs'/name/'report.json').read_text())
        parent = '/work/instance-nvme/runs/'+source.parent.name+'/checkpoint.pt'
        prior = 10368
        base = json.loads((a.root/'snapshots/mature-fp32-v1-ordinary-s103-u10368/launch.json').read_text())['arguments']
        for target in [17541, 19490]:
            name = f'{identity}-ordinary-s103-u{target}'
            args = base+['--resume', parent, '--steps', str(target-prior),
                '--warmup', '4', '--decay-start', str(count(17541)), '--decay-end', str(count(19490)),
                '--extend-stable-schedule', '--checkpoint-every', '1024']
            row = native(name, 'scripts.train_primitive_fp32', args)
            if (not row['all_replica_model_optimizer_states_equal'] or row['total_real_targets'] != count(target)
                    or row['total_training_matmul_flops'] != target*plan['ordinary_step_flops']):
                raise AssertionError('State/target/compute counters differ')
            parent = '/work/instance-nvme/runs/'+name+'/checkpoint.pt'
            prior = target
            result = native(name+'-dev-r1', 'scripts.evaluate_primitive_fp32', ['--checkpoint', parent,
                '--corpus', '/work/instance-nvme/variable-depth-fresh-v1/dev.json', '--split', 'dev',
                '--cohort', 'curve', '--repetitions', '1'], True)
            write_state(public/f'native-u{target}.json', result)
        audit.wait(timeout=max(1, deadline-time.time()-30))
        if audit.returncode: raise RuntimeError('Independent audit failed')
        frozen()
        write_state(public/'independent-cpu-verification.json', json.loads((output/'independent-cpu/report.json').read_text()))
        report = {'status': 'COMPLETE_ONE_PAIR_FP32_3B_COMPUTE_CONTROL', 'wall_seconds': time.time()-started,
                  'test_accessed': False, 'scope': plan['scope']}
    except BaseException as error:
        if active and active.poll() is None:
            subprocess.run(['docker', 'stop', '--time', '5', 'autonomous-compute-neuron233'], capture_output=True, timeout=25)
            try: active.wait(timeout=30)
            except subprocess.TimeoutExpired: active.terminate()
        if audit and audit.poll() is None:
            os.killpg(audit.pid, signal.SIGTERM)
            try: audit.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(audit.pid, signal.SIGKILL); audit.wait(timeout=10)
        report = {'status': 'STOPPED_ONE_PAIR_FP32_3B_COMPUTE_CONTROL', 'error': str(error),
                  'wall_seconds': time.time()-started, 'test_accessed': False}
        raise
    finally:
        write_state(output/'report.json', report)
        write_state(public/'report.json', report)
        write_state(public/'execution-costs.json', {'status': report['status'], 'attempts': costs,
            'wall_seconds': time.time()-started, 'scope': 'Half-node native assignments overlap other campaigns; CPU waiting is separate. Include failed/stopped attempts.'})


if __name__ == '__main__':
    main()
