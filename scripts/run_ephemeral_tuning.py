"""Finite trained-memory control; admission requires replication and qualification."""
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

ID = 'ephemeral-workspace-tuning-v1'
PUBLIC = Path('results/ephemeral_workspace_20261029/tuning-v1')
GRID = [.0003, .001, .003]


def admission_evidence(replication, qualification):
    if (replication.get('status') != 'COMPLETE_INTERLEAVED_SEED_REPLICATION'
            or replication.get('reproducibility_screen_pass') is not True):
        raise AssertionError('Complete positive frozen two-new-seed replication required')
    if qualification.get('status') != 'PASS_EPHEMERAL_ACTUAL_SHAPE_AND_REPLAY':
        raise AssertionError('Actual-shape gradient, optimizer and replay gates required')


def build_arms(old):
    from compute_trainium.model import ModelConfig
    from compute_trainium.ephemeral_model_factory import matrix_work
    config = ModelConfig(**old['model_config'])
    arms, points = {}, []
    for i, lr in enumerate(GRID):
        arm = copy.deepcopy(old['arms'][f'latent-lr{i}'])
        if arm['lr'] != lr or arm['checkpoint_updates'] != [1957]:
            raise AssertionError('Original three opportunities and complete endpoint required')
        name = f'ephemeral-lr{i}'
        arm.update(family='ephemeral', phase=ID+'-'+name,
                   workspace_mode='latent', depth_schedule=[0]*1957, audit_depths=[0],
                   nominal_matrix_flops=1957*64*matrix_work(
                       config, 512, 0, batch=old['batch'])['training_matmul_flops'])
        arms[name] = arm
        points.append(dict(label=name, arm=name, update=1957,
                           variants=[dict(label='main', depth=0, mode='latent')]))
    return arms, points


def prepare(root):
    replication = Path('results/interleaved_workspace_20261029/seed-replication-v1/report.json')
    qualification = Path('results/ephemeral_workspace_20261029/numerical-v1/report.json')
    admission_evidence(json.loads(replication.read_text()), json.loads(qualification.read_text()))
    if managed_running():
        raise RuntimeError('Existing Native owner must close')
    if shutil.disk_usage('/').free < 35*1024**3 or shutil.disk_usage(root).free < 39*1024**3:
        raise RuntimeError('Four GiB new artifacts plus protected disk reserves')
    qualified_path = root/'ephemeral-workspace-qualification-v1/plan.json'
    qualified = json.loads(qualified_path.read_text())
    scientific_files = ('compute_trainium/ephemeral_workspace.py',
                        'compute_trainium/ephemeral_model_factory.py',
                        'scripts/ephemeral_entry.py', 'compute_trainium/model.py',
                        'scripts/train_interleaved_screen.py',
                        'scripts/audit_interleaved_screen.py',
                        'scripts/evaluate_interleaved_screen.py')
    for f in scientific_files:
        if file_sha256(f) != qualified['source_sha256'][f]:
            raise AssertionError('Qualified graph or engine changed')
    old_path = root/'interleaved-lr-tuning-v1/plan.json'
    old = json.loads(old_path.read_text())
    arms, points = build_arms(old)
    if old['seed'] != 9901 or old['parameters'] != 51747328:
        raise AssertionError('Retain original paired initialization and parameter count')
    gates = {str(f): file_sha256(f) for f in (replication, qualification)}
    artifacts = [root/'interleaved-mixed-sequential-dev-v1/report.json',
                 *sorted((root/'interleaved-mixed-sequential-dev-v1').glob('PRIVATE-worker*-scores.json')),
                 root/'interleaved-tied-ordinary-tuning-v1/analysis/report.json']
    if len(artifacts) != 4:
        raise AssertionError('Both exact sequential score shards required')
    sources = {str(f): file_sha256(f) for d in ('compute_trainium', 'scripts', 'configs')
               for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh', 'docs/EPHEMERAL_TRAINED_CONTROL_20261029.md',
              'tests/test_ephemeral_tuning.py'):
        sources[f] = file_sha256(f)
    worker, host = split_snapshot_sources(sources)
    phases = [a['phase'] for a in arms.values()] + [ID+'-evaluation']
    projection = project_snapshot_inodes(Path('.'), len(phases))
    if os.statvfs(root).f_favail < projection['required_free_inodes']:
        raise RuntimeError('All four snapshots and protected inode reserve must fit')
    now = time.time()
    p = copy.deepcopy(old)
    p.update(identity=ID, registered_unix=now, seconds_limit=7200,
             absolute_deadline_unix=now+7200, phase_seconds_limit=1800,
             cpu_phase_seconds_limit=1200, arms=arms, evaluation_points=points,
             evaluation_panels=['broad'], source_sha256=sources,
             worker_source_sha256=worker, host_only_source_sha256=host,
             root_comm_ports=reserve_ports(root, phases), phases={n: {} for n in phases},
             inode_projection=projection, artifact_allowance_bytes=4*1024**3,
             workspace_mechanism='ephemeral_generated_state',
             ephemeral_training_graph='reused_source_kv',
             worker_entry='scripts.ephemeral_scientific_entry',
             gate_sha256=gates, qualification_plan_sha256=file_sha256(qualified_path),
             prior_plan_sha256=file_sha256(old_path),
             comparison_artifact_sha256={str(f.relative_to(root)): file_sha256(f) for f in artifacts},
             practical_gain_nats=.01, bootstrap_seed=40319, bootstrap_replicates=2000,
             automatic_promotion=False, protected_groups_scored=0, terminal_alpha_spent=0,
             scope='Single-seed independently tuned trained ephemeral-memory control at equal authentic data. Reused DEV; no confirmation, general reasoning, thought-budget scaling, allocation or matched-training-work claim.')
    out = root/ID
    out.mkdir()
    PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', p)
    write_state(PUBLIC/'plan.json', p)


def run(root):
    out = root/ID
    e = ScreenExecution(root, out/'plan.json', out/'execution')
    result = dict(status='RUNNING_EPHEMERAL_TRAINED_CONTROL', records={})
    try:
        for f, h in e.plan['gate_sha256'].items():
            if file_sha256(f) != h:
                raise AssertionError('Admission evidence changed')
        for arm, config in e.plan['arms'].items():
            name = config['phase']
            result['records'][arm] = e.native(name, 'scripts.ephemeral_scientific_entry',
                ['--entry', 'train', '--root', '/work/instance-nvme', '--plan',
                 f'/work/instance-nvme/{ID}/plan.json', '--plan-sha', e.sha, '--arm', arm,
                 '--comm-port', str(e.plan['root_comm_ports'][name])],
                'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS')
            write_state(PUBLIC/'progress.json', result)
            dest = out/('audit-'+arm)
            result['records']['audit-'+arm] = e.cpu(ID+'-audit-'+arm, 'scripts.ephemeral_scientific_entry',
                ['--entry', 'scientific-audit', '--root', str(root), '--plan', str(out/'plan.json'),
                 '--plan-sha', e.sha, '--arm', arm, '--output', str(dest)],
                dest/'report.json', 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS')
            write_state(PUBLIC/'progress.json', result)
        prior = json.loads(Path('results/interleaved_workspace_20261029/tuning-v1/report.json').read_text())
        expected = prior['records']['ordinary-lr0']['initial_model_digest']
        if any(result['records'][a]['initial_model_digest'] != expected for a in e.plan['arms']):
            raise AssertionError('Original same-shape random initialization required')
        phase = ID+'-evaluation'
        result['records']['evaluation'] = e.native(phase, 'scripts.ephemeral_scientific_entry',
            ['--entry', 'evaluate', '--root', '/work/instance-nvme', '--plan',
             f'/work/instance-nvme/{ID}/plan.json', '--plan-sha', e.sha,
             '--comm-port', str(e.plan['root_comm_ports'][phase])],
            'COMPLETE_INTERLEAVED_SCREEN_EVALUATION')
        dest = out/'analysis'
        result['records']['analysis'] = e.cpu(ID+'-analysis', 'scripts.analyze_ephemeral_tuning',
            ['--root', str(root), '--plan', str(out/'plan.json'), '--output', str(dest)],
            dest/'report.json', 'COMPLETE_EPHEMERAL_TRAINED_COMPARISON')
        write_state(PUBLIC/'analysis.json', result['records']['analysis'])
        result['status'] = 'COMPLETE_EPHEMERAL_TRAINED_CONTROL'
    except BaseException as error:
        result.update(status='STOPPED_EPHEMERAL_TRAINED_CONTROL', error_type=type(error).__name__, error=str(error))
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
