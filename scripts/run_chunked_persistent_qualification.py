"""Single bounded disposable feasibility test, after the prior fit closes."""
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
from scripts.run_probability_screen import ScreenExecution

ID = 'chunked-persistent-qualification-v1'
PUBLIC = Path('results/chunked_persistent_workspace_20261029/numerical-v1')


def prepare(root):
    prior_result = Path('results/depth_causal_workspace_20261029/fixed-one-v1/report.json')
    if json.loads(prior_result.read_text())['status'] != 'COMPLETE_FIXED_DEPTH_CAUSAL_COMPARISON':
        raise AssertionError('Complete and preserve the existing experiment first')
    if managed_running():
        raise RuntimeError('Existing Native owner must close')
    if shutil.disk_usage('/').free < 35*1024**3 or shutil.disk_usage(root).free < 39*1024**3:
        raise RuntimeError('Four GiB artifact allowance plus protected reserves')
    receipt = Path('results/chunked_persistent_workspace_20261029/cpu-v1/report.json')
    cpu = json.loads(receipt.read_text())
    if cpu['status'] != 'PASS_CPU_CHUNKED_PERSISTENT_REFERENCE':
        raise AssertionError('Complete CPU tests required')
    if any(file_sha256(f) != h for f, h in cpu['source_sha256'].items()):
        raise AssertionError('Tested source changed')
    p = copy.deepcopy(json.loads((root/'budgeted-interleaved-qualification-v1/plan.json').read_text()))
    from compute_trainium.model import ModelConfig
    ordinary = json.loads((root/'interleaved-tied-ordinary-tuning-v1/plan.json').read_text())
    p['model_config'] = ordinary['model_config']
    p['parameters'] = ModelConfig(**p['model_config']).parameter_count()
    p['untied_prediction_table'] = False
    p['source_stride'] = 128
    data = root/p['data_relative']
    data_report = json.loads((data/'report.json').read_text())
    if file_sha256(data/'report.json') != p['data_report_sha256'] or any(file_sha256(data/f) != h for f, h in data_report['files'].items()):
        raise AssertionError('Pinned disposable TRAIN bytes changed')
    sources = {str(f): file_sha256(f) for d in ('compute_trainium', 'scripts', 'configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh', 'tests/test_chunked_persistent_workspace.py',
              'docs/CHUNKED_PERSISTENT_WORKSPACE_REFERENCE_20261029.md', 'docs/CHUNKED_PERSISTENT_QUALIFICATION_20261029.md'):
        sources[f] = file_sha256(f)
    worker, host = split_snapshot_sources(sources)
    projection = project_snapshot_inodes(Path('.'), 2)
    if os.statvfs(root).f_favail < projection['required_free_inodes']:
        raise RuntimeError('Both snapshots must fit above protected inode reserve')
    phases = [ID+'-'+s for s in ('warm', 'replay')]
    now = time.time()
    p.update(identity=ID, registered_unix=now, absolute_deadline_unix=now+2700,
             seconds_limit=2700, phase_seconds_limit=1200, cpu_phase_seconds_limit=900,
             source_sha256=sources, worker_source_sha256=worker, host_only_source_sha256=host,
             root_comm_ports=reserve_ports(root, phases), phases={n: {} for n in phases},
             inode_projection=projection, artifact_allowance_bytes=4*1024**3,
             seed=40481, workspace_mode='registered', workspace_mechanism='chunked_persistent_generated_state',
             cold_local_supported_steps=[0, 1, 2, 3], depth_schedule=[0, 1, 2, 3],
             graph_specs={'0': 'full_ordinary', '1': 'chunked_ordinary', '2': 'generated_slot', '3': 'constant_slot'},
             replay_depth=2, warm_phase=phases[0], early_warm_mean_seconds_limit=2.,
             per_graph_repeated_update_seconds_limit=1., generated_to_full_ordinary_seconds_ratio_limit=3., worker_entry='scripts.chunked_persistent_entry',
             qualification_report_sha256=file_sha256(receipt), previous_learning_report_sha256=file_sha256(prior_result),
             qualification_only=True, scientific_fit_admitted=False, retained_for_science=False,
             scope='Sparse persistent actual-shape qualification only, using strongest tied ordinary shape and four explicit execution graphs. Exactly one slot per128 source tokens for generated/constant modes. No scientific fit, held-out source, router, confirmation or automatic promotion. All disposable states excluded from scientific initialization.')
    p.pop('previous_replication_screen_pass', None)
    out = root/ID
    out.mkdir()
    PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', p)
    write_state(PUBLIC/'plan.json', p)


def run(root):
    out = root/ID
    e = ScreenExecution(root, out/'plan.json', out/'execution')
    result = dict(status='RUNNING_CHUNKED_PERSISTENT_QUALIFICATION', records={})
    try:
        for stage in ('warm', 'replay'):
            name = ID+'-'+stage
            args = ['--entry', 'qualification', '--root', '/work/instance-nvme', '--plan',
                    f'/work/instance-nvme/{ID}/plan.json', '--plan-sha', e.sha, '--mode', stage,
                    '--comm-port', str(e.plan['root_comm_ports'][name])]
            if stage == 'replay':
                args += ['--parent', f'/work/instance-nvme/runs/{ID}-warm']
            expected = 'COMPLETE_INTERLEAVED_WARM_PENDING_GLOBAL_AUDIT' if stage == 'warm' else 'PASS_INTERLEAVED_WARM_EXACT_REPLAY'
            result['records'][stage] = e.native(name, 'scripts.chunked_persistent_entry', args, expected)
            if stage == 'warm':
                r = result['records'][stage]
                if r['warm_updates'] != 100 or set(r['warm_by_depth']) != {'0', '1', '2', '3'}:
                    raise AssertionError('All four graphs and100 balanced warm updates required')
                if max(r['warm_by_depth'].values()) > e.plan['per_graph_repeated_update_seconds_limit']:
                    raise AssertionError('Per-graph measured throughput gate failed')
                if r['warm_by_depth']['2']/r['warm_by_depth']['0'] > e.plan['generated_to_full_ordinary_seconds_ratio_limit']:
                    raise AssertionError('Generated slot exceeds the registered whole-update cost ceiling')
            write_state(PUBLIC/'progress.json', result)
        dest = out/'cpu-audit'
        result['records']['audit'] = e.cpu(ID+'-audit', 'scripts.chunked_persistent_entry',
            ['--entry', 'audit', '--root', str(root), '--plan', str(out/'plan.json'),
             '--plan-sha', e.sha, '--output', str(dest)], dest/'report.json', 'PASS_INTERLEAVED_MATURE_GLOBAL_CPU')
        result['status'] = 'PASS_CHUNKED_PERSISTENT_ACTUAL_SHAPE'
    except BaseException as error:
        result.update(status='STOPPED_CHUNKED_PERSISTENT_QUALIFICATION', error_type=type(error).__name__, error=str(error))
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
