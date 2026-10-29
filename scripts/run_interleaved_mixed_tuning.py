"""Separately admitted mixed-round follow-up; preserves original fixed-K2 fits."""
import argparse
import copy
import json
import os
import random
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

ID = 'interleaved-mixed-round-tuning-v1'
PUBLIC = Path('results/interleaved_workspace_20261029/mixed-round-tuning-v1')
BASE = 'interleaved-lr-tuning-v1'
GRID = [.0003, .001, .003]


def mixed_schedule(updates, seed=9917):
    if type(updates) is not int or updates < 2:
        raise ValueError('At least two updates required')
    schedule = [2]*((updates+1)//2) + [3]*(updates//2)
    random.Random(seed).shuffle(schedule)
    return schedule


def prepare(root):
    from compute_trainium.model import ModelConfig
    from compute_trainium.interleaved_workspace import matrix_work
    if managed_running():
        raise RuntimeError('Finish active Native owner before admission')
    prior = Path('results/interleaved_workspace_20261029/tuning-v1/report.json')
    if json.loads(prior.read_text())['status'] != 'COMPLETE_INTERLEAVED_LR_TUNING':
        raise AssertionError('Complete fixed-two comparison required')
    qualification = Path('results/interleaved_workspace_20261029/mixed-round-qualification-v1/report.json')
    q = json.loads(qualification.read_text())
    if q['status'] != 'PASS_INTERLEAVED_ACTUAL_SHAPE_AND_REPLAY':
        raise AssertionError('Both resident graphs and K3 replay must qualify')
    if shutil.disk_usage('/').free < 35*1024**3 or shutil.disk_usage(root).free < 39*1024**3:
        raise RuntimeError('Four-GiB allowance plus protected disk reserves')
    old = json.loads((root / BASE / 'plan.json').read_text())
    qualified = json.loads((root / 'interleaved-mixed-round-qualification-v1/plan.json').read_text())
    for f in ('compute_trainium/grouped_interleaved_workspace.py', 'compute_trainium/interleaved_workspace.py',
              'compute_trainium/model.py', 'scripts/train_interleaved_screen.py'):
        if file_sha256(f) != old['source_sha256'][f] or file_sha256(f) != qualified['source_sha256'][f]:
            raise AssertionError('Comparison must retain original qualified training computation')
    p = copy.deepcopy(old); c = ModelConfig(**p['model_config'])
    arms, points = {}, []
    for i, lr in enumerate(GRID):
        a = copy.deepcopy(old['arms'][f'latent-lr{i}']); name = f'mixed-lr{i}'
        schedule = mixed_schedule(len(a['depth_schedule']))
        a.update(family='latent', phase=ID+'-'+name, depth_schedule=schedule, audit_depths=[2, 3],
            nominal_matrix_flops=sum(64*matrix_work(c, 512, k, mode='latent', batch=p['batch'])['training_matmul_flops'] for k in schedule))
        arms[name] = a
        points.append(dict(label=name, arm=name, update=a['checkpoint_updates'][-1],
            variants=[dict(label='main', depth=2, mode='latent'), dict(label='supported3', depth=3, mode='latent')]))
    files = {str(f): file_sha256(f) for d in ('compute_trainium', 'scripts', 'configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh', 'docs/INTERLEAVED_MIXED_ROUND_TUNING_DRAFT_20261029.md'):
        files[f] = file_sha256(f)
    worker, host = split_snapshot_sources(files)
    names = [a['phase'] for a in arms.values()] + [ID+'-evaluation']
    projection = project_snapshot_inodes(Path('.'), len(names))
    if os.statvfs(root).f_favail < projection['required_free_inodes']:
        raise RuntimeError('Whole follow-up queue inode projection')
    now = time.time()
    p.update(identity=ID, registered_unix=now, seconds_limit=10800, absolute_deadline_unix=now+10800,
        source_sha256=files, worker_source_sha256=worker, host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root, names), phases={n:{} for n in names}, inode_projection=projection,
        arms=arms, evaluation_points=points, artifact_allowance_bytes=4*1024**3,
        prior_plan_sha256=file_sha256(root / BASE / 'plan.json'), prior_report_sha256=file_sha256(prior),
        qualification_report_sha256=file_sha256(qualification), depth_schedule_seed=9917,
        scope='Single-seed mixed K2/K3 numerical-depth follow-up at unchanged authentic source exposure. Reused controls and development panel, no thinking, efficiency or confirmation claim.')
    out = root / ID; out.mkdir(); PUBLIC.mkdir(parents=True)
    write_state(out / 'plan.json', p); write_state(PUBLIC / 'plan.json', p)
    print(json.dumps(dict(status='REGISTERED_MIXED_ROUND_TUNING', arms=len(arms), updates=len(schedule),
        k2_updates=schedule.count(2), k3_updates=schedule.count(3), inode_projection=projection)))


def run(root):
    out = root / ID; e = ScreenExecution(root, out / 'plan.json', out / 'execution')
    result = dict(status='RUNNING_MIXED_ROUND_TUNING', records={})
    try:
        for arm, a in e.plan['arms'].items():
            name = a['phase']
            args = ['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
                    '--plan-sha',e.sha,'--arm',arm,'--comm-port',str(e.plan['root_comm_ports'][name])]
            result['records'][arm] = e.native(name, 'scripts.train_interleaved_screen', args, 'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS')
            write_state(PUBLIC / 'progress.json', result)
            dest = out / ('audit-'+arm)
            result['records']['audit-'+arm] = e.cpu(ID+'-audit-'+arm, 'scripts.audit_interleaved_screen',
                ['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--arm',arm,'--output',str(dest)],
                dest/'report.json', 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS')
            write_state(PUBLIC / 'progress.json', result)
        prior = json.loads(Path('results/interleaved_workspace_20261029/tuning-v1/report.json').read_text())
        expected = prior['records']['ordinary-lr0']['initial_model_digest']
        if any(result['records'][arm]['initial_model_digest'] != expected for arm in e.plan['arms']):
            raise AssertionError('Original paired initialization required')
        name = ID+'-evaluation'
        result['records']['evaluation'] = e.native(name, 'scripts.evaluate_interleaved_screen',
            ['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,
             '--comm-port',str(e.plan['root_comm_ports'][name])], 'COMPLETE_INTERLEAVED_SCREEN_EVALUATION')
        dest = out / 'analysis'
        result['records']['analysis'] = e.cpu(ID+'-analysis', 'scripts.analyze_interleaved_mixed_tuning',
            ['--root',str(root),'--plan',str(out/'plan.json'),'--output',str(dest)],
            dest/'report.json', 'COMPLETE_MIXED_ROUND_TUNING_ANALYSIS')
        write_state(PUBLIC / 'analysis.json', result['records']['analysis'])
        result['status'] = 'COMPLETE_MIXED_ROUND_TUNING'
    except BaseException as error:
        result.update(status='STOPPED_MIXED_ROUND_TUNING', error_type=type(error).__name__, error=str(error)); raise
    finally:
        write_state(PUBLIC / 'report.json', result); e.close(PUBLIC / 'execution-costs.json')


if __name__ == '__main__':
    a=argparse.ArgumentParser(); a.add_argument('--root',type=Path,required=True)
    a.add_argument('--action',choices=['prepare','run'],required=True); p=a.parse_args()
    (prepare if p.action=='prepare' else run)(p.root)
