"""One bounded numerical test of actual latent-token budgets, without a fit."""
import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from compute_trainium.budgeted_interleaved_factory import GRAPH_SPECS
from scripts.run_probability_screen import ScreenExecution

ID = 'budgeted-interleaved-qualification-v1'
PUBLIC = Path('results/budgeted_interleaved_20261029/numerical-v1')


def timing_summary(updates):
    if [r['update'] for r in updates] != list(range(1, 105)):
        raise AssertionError('All104 disposable updates must complete')
    seen, first, remaining = set(), [], []
    for row in updates:
        key = row['extra_steps']
        if key not in GRAPH_SPECS:
            raise AssertionError('Unregistered graph')
        if key not in seen:
            first.append(dict(update=row['update'], graph_id=key, seconds=row['seconds']))
            seen.add(key)
        else:
            remaining.append(row)
    if seen != set(GRAPH_SPECS):
        raise AssertionError('All five numerical graphs must execute')
    by_graph = {str(k): dict(updates=sum(r['extra_steps'] == k for r in remaining),
                            mean_seconds=sum(r['seconds'] for r in remaining if r['extra_steps'] == k)
                            /sum(r['extra_steps'] == k for r in remaining)) for k in GRAPH_SPECS}
    if any(r['updates'] < 18 for r in by_graph.values()):
        raise AssertionError('At least18 measured repeat updates per graph')
    return dict(first_graph_uses=first, remaining_updates=len(remaining), by_graph=by_graph,
                throughput_pass=all(r['mean_seconds'] <= .8 for r in by_graph.values()))


def prepare(root):
    rep_path = Path('results/interleaved_workspace_20261029/seed-replication-v1/report.json')
    rep = json.loads(rep_path.read_text())
    if rep['status'] != 'COMPLETE_INTERLEAVED_SEED_REPLICATION':
        raise AssertionError('Close the frozen replication before a new mechanism test')
    if managed_running():
        raise RuntimeError('Existing Native owner must close')
    if min(shutil.disk_usage('/').free, shutil.disk_usage(root).free) < 35*1024**3 or shutil.disk_usage(root).free < 39*1024**3:
        raise RuntimeError('Four GiB qualification allowance plus protected reserves')
    cpu_path = Path('results/budgeted_interleaved_20261029/cpu-v1/report.json')
    cpu = json.loads(cpu_path.read_text())
    if cpu['status'] != 'PASS_BUDGETED_INTERLEAVED_CPU_REFERENCE':
        raise AssertionError('CPU graph, gradient and cost qualification first')
    if any(file_sha256(f) != h for f, h in cpu['source_sha256'].items()):
        raise AssertionError('Tested implementation changed')
    prior = root/'ephemeral-workspace-qualification-v1/plan.json'
    p = copy.deepcopy(json.loads(prior.read_text()))
    data = root/p['data_relative']
    data_report = json.loads((data/'report.json').read_text())
    if file_sha256(data/'report.json') != p['data_report_sha256'] or any(file_sha256(data/n) != h for n, h in data_report['files'].items()):
        raise AssertionError('Pinned disposable TRAIN bytes changed')
    sources = {str(f): file_sha256(f) for d in ('compute_trainium', 'scripts', 'configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh', 'docs/BUDGETED_INTERLEAVED_QUALIFICATION_20261029.md',
              'tests/test_budgeted_interleaved_reference.py', 'tests/test_budgeted_interleaved_factory.py'):
        sources[f] = file_sha256(f)
    worker, host = split_snapshot_sources(sources)
    projection = project_snapshot_inodes(Path('.'), 2)
    if os.statvfs(root).f_favail < projection['required_free_inodes']:
        raise RuntimeError('Both snapshots and protected inode reserve must fit')
    phases = [ID+'-'+stage for stage in ('warm', 'replay')]
    now = time.time()
    p.update(identity=ID, registered_unix=now, absolute_deadline_unix=now+3600,
             seconds_limit=3600, phase_seconds_limit=1800, cpu_phase_seconds_limit=900,
             source_sha256=sources, worker_source_sha256=worker, host_only_source_sha256=host,
             root_comm_ports=reserve_ports(root, phases), phases={n: {} for n in phases},
             inode_projection=projection, seed=40341, workspace_mode='latent',
             workspace_mechanism='budgeted_persistent_generated_state',
             cold_local_supported_steps=list(GRAPH_SPECS), depth_schedule=list(GRAPH_SPECS),
             graph_specs={str(k): dict(thought_budget=b, numerical_refinements=j) for k, (b, j) in GRAPH_SPECS.items()},
             replay_depth=25, warm_phase=phases[0], early_warm_mean_seconds_limit=60.,
             per_graph_repeated_update_seconds_limit=.8,
             worker_entry='scripts.budgeted_interleaved_entry',
             qualification_report_sha256=file_sha256(cpu_path),
             previous_replication_report_sha256=file_sha256(rep_path),
             previous_replication_screen_pass=rep['reproducibility_screen_pass'],
             scope='Disposable supported-budget qualification. The prior fixed-one replication result remains unchanged and gates its own prepared follow-up. This tests numerical feasibility for a different within-checkpoint question, not a scientific fit, scaling admission or fresh confirmation.')
    p.pop('ephemeral_training_graph', None)
    p.pop('graph_layout', None)
    out = root/ID
    out.mkdir()
    PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', p)
    write_state(PUBLIC/'plan.json', p)


def run(root):
    out = root/ID
    e = ScreenExecution(root, out/'plan.json', out/'execution')
    result = dict(status='RUNNING_BUDGETED_INTERLEAVED_QUALIFICATION', records={})
    try:
        for stage in ('warm', 'replay'):
            name = ID+'-'+stage
            args = ['--entry', 'qualification', '--root', '/work/instance-nvme', '--plan',
                    f'/work/instance-nvme/{ID}/plan.json', '--plan-sha', e.sha,
                    '--mode', stage, '--comm-port', str(e.plan['root_comm_ports'][name])]
            if stage == 'replay':
                args += ['--parent', f'/work/instance-nvme/runs/{ID}-warm']
            expected = 'COMPLETE_INTERLEAVED_WARM_PENDING_GLOBAL_AUDIT' if stage == 'warm' else 'PASS_INTERLEAVED_WARM_EXACT_REPLAY'
            result['records'][stage] = e.native(name, 'scripts.budgeted_interleaved_entry', args, expected)
            if stage == 'warm':
                updates = json.loads((root/'runs'/name/'updates.json').read_text())
                result['timing'] = timing_summary(updates)
                if not result['timing']['throughput_pass']:
                    raise AssertionError('Registered per-graph repeated-update ceiling failed')
            write_state(PUBLIC/'progress.json', result)
        dest = out/'cpu-audit'
        result['records']['audit'] = e.cpu(ID+'-audit', 'scripts.budgeted_interleaved_entry',
            ['--entry', 'audit', '--root', str(root), '--plan', str(out/'plan.json'),
             '--plan-sha', e.sha, '--output', str(dest)], dest/'report.json', 'PASS_INTERLEAVED_MATURE_GLOBAL_CPU')
        result['status'] = 'PASS_BUDGETED_INTERLEAVED_ACTUAL_SHAPE'
    except BaseException as error:
        result.update(status='STOPPED_BUDGETED_INTERLEAVED_QUALIFICATION', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_state(out/'report.json', result)
        write_state(PUBLIC/'report.json', result)
        e.close(PUBLIC/'execution-costs.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--action', choices=['prepare', 'run'], required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.root)
