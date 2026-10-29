"""Finite supported-budget discovery; no retry of the failed fixed-one gate."""
import argparse
import copy
import json
import os
import random
import signal
import shutil
import time
from pathlib import Path

from compute_trainium.budgeted_interleaved_factory import GRAPH_SPECS, matrix_work
from compute_trainium.execution_isolation import managed_running
from compute_trainium.model import ModelConfig
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution

ID = 'supported-budget-tuning-v1'
PUBLIC = Path('results/budgeted_interleaved_20261029/learning-v1')
GRID = [.0003, .001, .003]
SCHEDULE_SEED = 40379


def schedule():
    counts = {0: 653, 12: 326, 13: 326, 24: 326, 25: 326}
    prefix = list(GRAPH_SPECS)
    remaining = [k for k in prefix for _ in range(counts[k]-1)]
    random.Random(SCHEDULE_SEED).shuffle(remaining)
    return prefix+remaining


def build_arms(old):
    c = ModelConfig(**old['model_config'])
    steps = schedule()
    arms, points = {}, []
    for i, lr in enumerate(GRID):
        a = copy.deepcopy(old['arms'][f'latent-lr{i}'])
        if a['lr'] != lr or a['checkpoint_updates'] != [1957] or len(steps) != 1957:
            raise AssertionError('Original full endpoint and LR opportunities required')
        name = f'budget-lr{i}'
        a.update(family='supported_budget', phase=ID+'-'+name, depth_schedule=steps,
                 audit_depths=list(GRAPH_SPECS), workspace_mode='latent',
                 nominal_matrix_flops=sum(64*matrix_work(c, 512, k, batch=old['batch'])['training_matmul_flops'] for k in steps))
        arms[name] = a
        points.append(dict(label=name, arm=name, update=1957, variants=[
            dict(label=f'b{b}', depth=k, mode='latent') for b, k in [(0, 0), (1, 13), (2, 25)]]))
    return arms, points


def prepare(root):
    qualification = Path('results/budgeted_interleaved_20261029/numerical-v1/report.json')
    replication = Path('results/interleaved_workspace_20261029/seed-replication-v1/report.json')
    if json.loads(qualification.read_text())['status'] != 'PASS_BUDGETED_INTERLEAVED_ACTUAL_SHAPE':
        raise AssertionError('All-budget actual-shape qualification required')
    rep = json.loads(replication.read_text())
    if rep['status'] != 'COMPLETE_INTERLEAVED_SEED_REPLICATION':
        raise AssertionError('Close and retain previous replication first')
    if managed_running(): raise RuntimeError('Previous Native owner must close')
    if shutil.disk_usage('/').free < 35*1024**3 or shutil.disk_usage(root).free < 41*1024**3:
        raise RuntimeError('Six GiB artifacts plus protected reserves')
    qpath = root/'budgeted-interleaved-qualification-v1/plan.json'
    q = json.loads(qpath.read_text())
    for f in ('compute_trainium/budgeted_interleaved_reference.py',
              'compute_trainium/budgeted_interleaved_factory.py',
              'compute_trainium/model.py', 'scripts/train_interleaved_screen.py',
              'scripts/audit_interleaved_screen.py', 'scripts/evaluate_interleaved_screen.py'):
        if file_sha256(f) != q['source_sha256'][f]:
            raise AssertionError('Qualified scientific implementation changed')
    old_path = root/'interleaved-lr-tuning-v1/plan.json'
    old = json.loads(old_path.read_text())
    evaluation = root/old['evaluation_relative']
    evaluation_files = json.loads((evaluation/'report.json').read_text())['files']
    if file_sha256(evaluation/'report.json') != old['evaluation_report_sha256'] or any(
            file_sha256(evaluation/f) != h for f, h in evaluation_files.items()):
        raise AssertionError('Frozen original evaluation bytes changed')
    arms, points = build_arms(old)
    consistency_path = root/'interleaved-convergence-v1/plan.json'
    consistency = json.loads(consistency_path.read_text())
    artifacts = [root/'interleaved-mixed-sequential-dev-v1/report.json',
        *sorted((root/'interleaved-mixed-sequential-dev-v1').glob('PRIVATE-worker*-scores.json')),
        root/'interleaved-tied-ordinary-tuning-v1/analysis/report.json',
        Path('results/interleaved_workspace_20261029/tuning-v1/analysis.json')]
    # Include all older score shards whose frozen ordinary/pause endpoints may
    # be compared later. None is selected again by this experiment.
    for trial in ('interleaved-tied-ordinary-tuning-v1', 'interleaved-lr-tuning-v1'):
        artifacts += [root/'runs'/(trial+'-evaluation')/f'PRIVATE-rank{k}-scores.json' for k in range(64)]
    sources = {str(f): file_sha256(f) for d in ('compute_trainium', 'scripts', 'configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh', 'docs/SUPPORTED_LATENT_BUDGET_LEARNING_20261029.md',
              'tests/test_supported_budget_tuning.py'):
        sources[f] = file_sha256(f)
    worker, host = split_snapshot_sources(sources)
    phases = [a['phase'] for a in arms.values()]+[ID+'-evaluation']
    projection = project_snapshot_inodes(Path('.'), len(phases))
    if os.statvfs(root).f_favail < projection['required_free_inodes']:
        raise RuntimeError('Every snapshot must fit above protected inode reserve')
    now = time.time()
    p = copy.deepcopy(old)
    p.update(identity=ID, registered_unix=now, seconds_limit=10800, absolute_deadline_unix=now+10800,
             phase_seconds_limit=1800, cpu_phase_seconds_limit=1800,
             source_sha256=sources, worker_source_sha256=worker, host_only_source_sha256=host,
             root_comm_ports=reserve_ports(root, phases), phases={n: {} for n in phases},
             inode_projection=projection, artifact_allowance_bytes=6*1024**3,
             arms=arms, evaluation_points=points, evaluation_panels=['broad'],
             evaluation_files=evaluation_files,
             workspace_mechanism='budgeted_persistent_generated_state',
             graph_specs={str(k): dict(thought_budget=b, numerical_refinements=j) for k,(b,j) in GRAPH_SPECS.items()},
             graph_schedule_seed=SCHEDULE_SEED, worker_entry='scripts.supported_budget_entry',
             consistency_windows=consistency['windows'], consistency_guard=consistency['guard'],
             consistency_rounds=[0, 1, 2, 3, 4, 5, 8, 16, 32],
             consistency_parent_plan_sha256=file_sha256(consistency_path),
             qualification_plan_sha256=file_sha256(qpath),
             gate_sha256={str(f): file_sha256(f) for f in (qualification, replication)},
             previous_fixed_one_reproducibility_screen=rep['reproducibility_screen_pass'],
             comparison_root_sha256={str(f.relative_to(root)): file_sha256(f) for f in artifacts if f.is_absolute()},
             comparison_repo_sha256={str(f): file_sha256(f) for f in artifacts if not f.is_absolute()},
             practical_gain_nats=.01, zero_exit_noninferiority_margin=.005,
             bootstrap_seed=40387, bootstrap_replicates=2000,
             exact_readout_workers=4, exact_readout_threads=8, exact_readout_seconds_limit=1500,
             protected_groups_scored=0, terminal_alpha_spent=0, automatic_promotion=False,
             scope='New single-seed supported zero/one/two-latent discovery, equal authentic data. The fixed-one replication failure is retained. Reused DEV; no content-specific second-token, allocation, efficiency or confirmation claim.')
    out = root/ID
    out.mkdir()
    PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', p)
    write_state(PUBLIC/'plan.json', p)


def run(root):
    out = root/ID
    e = ScreenExecution(root, out/'plan.json', out/'execution')
    result = dict(status='RUNNING_SUPPORTED_BUDGET_LEARNING', records={})
    def stop(number, frame):
        raise KeyboardInterrupt('Supported-budget owner interrupted')
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for f,h in e.plan['gate_sha256'].items():
            if file_sha256(f) != h: raise AssertionError('Admission evidence changed')
        for arm, a in e.plan['arms'].items():
            name = a['phase']
            result['records'][arm] = e.native(name, 'scripts.supported_budget_entry',
                ['--entry', 'train', '--root', '/work/instance-nvme', '--plan', f'/work/instance-nvme/{ID}/plan.json',
                 '--plan-sha', e.sha, '--arm', arm, '--comm-port', str(e.plan['root_comm_ports'][name])],
                'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS')
            write_state(PUBLIC/'progress.json', result)
            dest = out/('audit-'+arm)
            result['records']['audit-'+arm] = e.cpu(ID+'-audit-'+arm, 'scripts.supported_budget_entry',
                ['--entry', 'scientific-audit', '--root', str(root), '--plan', str(out/'plan.json'),
                 '--plan-sha', e.sha, '--arm', arm, '--output', str(dest)], dest/'report.json',
                'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS')
            dest = out/('consistency-'+arm)
            result['records']['consistency-'+arm] = e.cpu(ID+'-consistency-'+arm, 'scripts.check_supported_budget_consistency',
                ['--root', str(root), '--plan', str(out/'plan.json'), '--plan-sha', e.sha,
                 '--arm', arm, '--output', str(dest)], dest/'report.json', 'PASS_SUPPORTED_BUDGET_TRAIN_CONSISTENCY')
            write_state(PUBLIC/'progress.json', result)
        old = json.loads(Path('results/interleaved_workspace_20261029/tuning-v1/report.json').read_text())
        if any(result['records'][a]['initial_model_digest'] != old['records']['ordinary-lr0']['initial_model_digest'] for a in e.plan['arms']):
            raise AssertionError('Original paired random initialization must match')
        name = ID+'-evaluation'
        result['records']['evaluation'] = e.native(name, 'scripts.supported_budget_entry',
            ['--entry', 'evaluate', '--root', '/work/instance-nvme', '--plan', f'/work/instance-nvme/{ID}/plan.json',
             '--plan-sha', e.sha, '--comm-port', str(e.plan['root_comm_ports'][name])],
            'COMPLETE_INTERLEAVED_SCREEN_EVALUATION')
        for action, expected in [('select', 'COMPLETE_SUPPORTED_BUDGET_LR_SELECTION'),
                                 ('readout', 'COMPLETE_SUPPORTED_BUDGET_EXACT_READOUT'),
                                 ('compare', 'COMPLETE_SUPPORTED_BUDGET_COMPARISON')]:
            dest = out/action
            module = 'scripts.readout_supported_budget' if action == 'readout' else 'scripts.analyze_supported_budget'
            args = ['--root', str(root), '--plan', str(out/'plan.json'), '--output', str(dest)]
            if action != 'readout': args += ['--action', action]
            result['records'][action] = e.cpu(ID+'-'+action, module, args, dest/'report.json', expected)
            write_state(PUBLIC/(action+'.json'), result['records'][action])
        result['status'] = 'COMPLETE_SUPPORTED_BUDGET_LEARNING'
    except BaseException as error:
        result.update(status='STOPPED_SUPPORTED_BUDGET_LEARNING', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_state(out/'report.json', result)
        write_state(PUBLIC/'report.json', result)
        e.close(PUBLIC/'execution-costs.json')
        for s, handler in handlers.items():
            signal.signal(s, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--action', choices=['prepare', 'run'], required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.root)
