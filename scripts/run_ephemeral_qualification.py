"""One bounded disposable qualification, after all active replication fits audit."""
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

ID = 'ephemeral-workspace-qualification-v1'
PUBLIC = Path('results/ephemeral_workspace_20261029/numerical-v1')


def completed_replication_fits(root):
    records = {}
    for seed in (730201, 730237):
        for role in ('latent', 'ordinary', 'pause'):
            f = root/f'interleaved-seed-replication-v1-s{seed}-{role}'/'report.json'
            r = json.loads(f.read_text())
            if r.get('status') != 'COMPLETE_AUDITED_REPLICATION_FIT':
                raise AssertionError('All six frozen replication fits must audit first')
            if r['audit']['status'] != 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':
                raise AssertionError('No failed numerical gate may be bypassed')
            records[str(f.relative_to(root))] = file_sha256(f)
    return records


def prepare(root):
    gates = completed_replication_fits(root)
    if managed_running():
        raise RuntimeError('Existing Native owner must close')
    if shutil.disk_usage('/').free < 35*1024**3 or shutil.disk_usage(root).free < 39*1024**3:
        raise RuntimeError('Four GiB qualification allowance plus protected reserves')
    projection = project_snapshot_inodes(Path('.'), 2)
    if os.statvfs(root).f_favail < projection['required_free_inodes']:
        raise RuntimeError('Both snapshots and protected inode reserve must fit')
    receipt = Path('results/ephemeral_workspace_20261029/cpu-v1/report.json')
    cpu = json.loads(receipt.read_text())
    if cpu['status'] != 'PASS_EPHEMERAL_CPU_REFERENCE':
        raise AssertionError('Complete CPU reference checks required')
    for f, h in cpu['source_sha256'].items():
        if file_sha256(f) != h:
            raise AssertionError('Tested CPU implementation changed')
    parent = root/'interleaved-mixed-round-qualification-v1/plan.json'
    p = copy.deepcopy(json.loads(parent.read_text()))
    data = root/p['data_relative']
    data_report = json.loads((data/'report.json').read_text())
    if file_sha256(data/'report.json') != p['data_report_sha256']:
        raise AssertionError('Pinned TRAIN qualification source changed')
    if any(file_sha256(data/n) != h for n, h in data_report['files'].items()):
        raise AssertionError('Pinned TRAIN bytes changed')
    files = {str(f): file_sha256(f) for d in ('compute_trainium', 'scripts', 'configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh', 'docs/EPHEMERAL_WORKSPACE_QUALIFICATION_20261029.md',
              'tests/test_ephemeral_workspace.py'):
        files[f] = file_sha256(f)
    worker, host = split_snapshot_sources(files)
    names = [ID+'-'+stage for stage in ('warm', 'replay')]
    now = time.time()
    p.update(identity=ID, registered_unix=now, absolute_deadline_unix=now+2700,
             seconds_limit=2700, phase_seconds_limit=1200, cpu_phase_seconds_limit=600,
             source_sha256=files, worker_source_sha256=worker, host_only_source_sha256=host,
             root_comm_ports=reserve_ports(root, names), phases={n: {} for n in names},
             inode_projection=projection, warm_phase=names[0], seed=40271,
             workspace_mode='latent', cold_local_supported_steps=[0], depth_schedule=[0], replay_depth=0,
             workspace_mechanism='ephemeral_generated_state', ephemeral_training_graph='reused_source_kv',
             qualification_report_sha256=file_sha256(receipt), prior_replication_fit_sha256=gates,
             worker_entry='scripts.ephemeral_entry',
             graph_layout='Ordinary source pass; parallel latent pass reads source KV and its own slot only. Full gradients through source KV. CPU reference recomputes source and latent using an independent expanded attention mask.',
             scope='Disposable actual-shape numerical/state/throughput qualification only. No scientific fit, evaluation source, retained scientific initialization or automatic promotion. Prior six replication fits must all pass; their predictive outcomes are not an admission input.')
    out = root/ID
    out.mkdir()
    PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', p)
    write_state(PUBLIC/'plan.json', p)


def run(root):
    out = root/ID
    e = ScreenExecution(root, out/'plan.json', out/'execution')
    result = dict(status='RUNNING_EPHEMERAL_QUALIFICATION', records={})
    try:
        for stage in ('warm', 'replay'):
            name = ID+'-'+stage
            args = ['--entry', 'qualification', '--root', '/work/instance-nvme',
                    '--plan', f'/work/instance-nvme/{ID}/plan.json', '--plan-sha', e.sha,
                    '--mode', stage, '--comm-port', str(e.plan['root_comm_ports'][name])]
            if stage == 'replay':
                args += ['--parent', f'/work/instance-nvme/runs/{ID}-warm']
            expected = ('COMPLETE_INTERLEAVED_WARM_PENDING_GLOBAL_AUDIT' if stage == 'warm'
                        else 'PASS_INTERLEAVED_WARM_EXACT_REPLAY')
            result['records'][stage] = e.native(name, 'scripts.ephemeral_entry', args, expected)
            write_state(PUBLIC/'progress.json', result)
        dest = out/'cpu-audit'
        result['records']['audit'] = e.cpu(ID+'-audit', 'scripts.ephemeral_entry',
            ['--entry', 'audit', '--root', str(root), '--plan', str(out/'plan.json'),
             '--plan-sha', e.sha, '--output', str(dest)], dest/'report.json', 'PASS_INTERLEAVED_MATURE_GLOBAL_CPU')
        result['status'] = 'PASS_EPHEMERAL_ACTUAL_SHAPE_AND_REPLAY'
    except BaseException as error:
        result.update(status='STOPPED_EPHEMERAL_QUALIFICATION', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_state(PUBLIC/'report.json', result)
        e.close(PUBLIC/'execution-costs.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--action', choices=['prepare', 'run'], required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.root)
