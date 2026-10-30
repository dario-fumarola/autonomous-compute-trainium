"""Recover the unstarted conditional control; preserve science and deadline."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_latent_exposure_screen as exposure

ID='latent-exposure-work-recovery-v1'
PUBLIC=Path('results/latent_exposure_20261030/work-recovery-v1')
RUNTIME_KEYS=('identity','registered_unix','seconds_limit','source_sha256',
    'worker_source_sha256','host_only_source_sha256','root_comm_ports','phases',
    'inode_projection','recovery')


def read(path):return json.loads(path.read_text())


def science(plan):
    value=copy.deepcopy(plan)
    for key in RUNTIME_KEYS:value.pop(key,None)
    for arm in value['arms'].values():arm.pop('phase',None)
    return value


def recovered_plan(original,now,sources,ports,projection):
    from compute_trainium.native_snapshot_manifest import split_snapshot_sources
    if now>=original['absolute_deadline_unix']:raise TimeoutError('Original deadline expired')
    phase=ID+'-fit';worker,host=split_snapshot_sources(sources)
    p=copy.deepcopy(original)
    p.update(identity=ID,registered_unix=now,
        seconds_limit=math.floor(original['absolute_deadline_unix']-now),
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=ports,phases={phase:{}},inode_projection=projection,
        recovery='Launch-name repair only; original zero-update failure and absolute deadline retained.')
    p['arms']['ordinary']['phase']=phase
    if science(p)!=science(original):raise AssertionError('Scientific recipe changed')
    return p


def prepare(root):
    from compute_trainium.execution_isolation import managed_running
    from compute_trainium.native_comm_ports import reserve_ports
    from compute_trainium.native_inode_projection import project_snapshot_inodes
    if managed_running():raise RuntimeError('Exclusive Native ownership required')
    out=root/ID
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every recovery attempt')
    old=root/exposure.ID;parent=read(old/'plan.json');result=read(old/'report.json')
    if (result['status']!='STOPPED_BOUNDED_LATENT_EXPOSURE'
        or result.get('error')!='Explicit safe all-chip identity required'
        or not result.get('equal_data_development_gate')
        or not exposure.advancement(result['equal_data']['broad'],result['equal_data']['arithmetic'])):
        raise AssertionError('Unchanged completed gate and exact initiating failure required')
    trial=parent['trials']['ordinary_work'];original_path=root/trial['identity']/'plan.json'
    if file_sha256(original_path)!=trial['plan_sha256']:raise AssertionError('Original work plan changed')
    original=read(original_path);costs=read(root/trial['identity']/'execution-costs.json')
    if costs['native_attempts'] or costs['current_controller_cpu_attempts']:
        raise AssertionError('This recovery admits no partial-history retry')
    if (root/'runs'/original['arms']['ordinary']['phase']).exists():
        raise AssertionError('Original control must be unstarted')
    for role in ('ordinary','pause','latent'):
        if result['records'][role]['status']!='COMPLETE_AUDITED_EXPOSURE_FIT':
            raise AssertionError('All retained endpoints must be audited')
    now=time.time();deadline=parent['absolute_deadline_unix']
    ordinary=result['records']['ordinary']['fit']
    projection_seconds=ordinary['mean_warm_update_seconds']*len(original['arms']['ordinary']['depth_schedule'])+300+1800+3600+1800
    if now+projection_seconds>deadline:raise TimeoutError('Measured remaining work does not fit original cap')
    if min(shutil.disk_usage(root).free-2*1024**3,shutil.disk_usage('/').free)<35*1024**3:
        raise RuntimeError('Preserve disk reserve plus remaining2GiB artifacts')
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Snapshot inode reserve')
    changed=[f for f,h in parent['source_sha256'].items() if file_sha256(f)!=h]
    if set(changed)!={'scripts/run_latent_exposure_screen.py','tests/test_latent_exposure_screen.py'}:
        raise AssertionError('Only the host launch-name helper and its regression test may change')
    sources={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/LATENT_LEARNING_EXPOSURE_20261030.md',
        'docs/LATENT_WORK_RECOVERY_20261030.md','tests/test_latent_exposure_screen.py',
        'tests/test_latent_work_recovery.py'):sources[f]=file_sha256(f)
    ports=reserve_ports(root,[ID+'-fit'])
    p=recovered_plan(original,now,sources,ports,projection)
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    readout_parent=copy.deepcopy(parent);readout_parent['source_sha256']=sources
    readout_parent['trials']['ordinary_work']=dict(identity=ID,role='ordinary',tied=True,
        seed=parent['seed'],plan_sha256=file_sha256(out/'plan.json'))
    manifest=dict(status='REGISTERED_LAUNCH_ONLY_WORK_RECOVERY',registered_unix=now,
        absolute_deadline_unix=deadline,projected_remaining_seconds=projection_seconds,
        preparation_wall_seconds=max(0.,now-costs['execution']['started_unix']-costs['wall_seconds_since_admission']),
        preparation_opportunity_chip_hours=16*max(0.,now-costs['execution']['started_unix']-costs['wall_seconds_since_admission'])/3600,
        preparation_assigned_chip_hours=0,
        original_parent_sha256=file_sha256(old/'plan.json'),original_result_sha256=file_sha256(old/'report.json'),
        original_work_plan_sha256=file_sha256(original_path),original_failed_costs_sha256=file_sha256(root/trial['identity']/'execution-costs.json'),
        plan_sha256=file_sha256(out/'plan.json'),readout_parent=readout_parent,
        scientific_payload_unchanged=science(p)==science(original),assigned_training_updates_before_recovery=0,
        protected_groups_scored=0,terminal_alpha_spent=0,scope=Path('docs/LATENT_WORK_RECOVERY_20261030.md').read_text())
    write_state(out/'manifest.json',manifest);write_state(PUBLIC/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('status','absolute_deadline_unix','projected_remaining_seconds','scientific_payload_unchanged')}),flush=True)


def run(root):
    from scripts.run_probability_screen import ScreenExecution
    out=root/ID;p=read(out/'plan.json');m=read(out/'manifest.json')
    if file_sha256(out/'plan.json')!=m['plan_sha256']:raise AssertionError('Recovery plan changed')
    if file_sha256(root/exposure.ID/'report.json')!=m['original_result_sha256']:
        raise AssertionError('Preserve original failure and completed quality results')
    e=ScreenExecution(root,out/'plan.json',out/'execution');start=time.time()
    result=dict(status='RUNNING_LATENT_WORK_RECOVERY',records={})
    try:
        phase=p['arms']['ordinary']['phase']
        result['records']['fit']=e.native(phase,'scripts.tied_interleaved_entry',[
            '--entry','train','--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
            '--plan-sha',e.sha,'--arm','ordinary','--comm-port',str(p['root_comm_ports'][phase])],
            'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS')
        write_state(PUBLIC/'progress.json',result)
        dest=out/'audit-ordinary'
        result['records']['audit']=e.cpu(ID+'-audit','scripts.tied_interleaved_entry',[
            '--entry','scientific-audit','--root',str(root),'--plan',str(out/'plan.json'),
            '--plan-sha',e.sha,'--arm','ordinary','--output',str(dest)],
            dest/'report.json','PASS_INTERLEAVED_SCREEN_MATURE_AUDITS')
        write_state(PUBLIC/'progress.json',result)
        exposure.PUBLIC=PUBLIC
        result['matched_work']=exposure.readouts(root,m['readout_parent'],'matched-work-recovery-v1','ordinary_work')
        result['matched_work_development_gate']=exposure.advancement(result['matched_work']['broad'],result['matched_work']['arithmetic'])
        result['status']='COMPLETE_LATENT_WORK_RECOVERY'
    except BaseException as error:
        result.update(status='STOPPED_LATENT_WORK_RECOVERY',error_type=type(error).__name__,error=str(error));raise
    finally:
        e.close(out/'execution-costs.json');write_state(PUBLIC/'execution-costs.json',read(out/'execution-costs.json'))
        result.update(wall_seconds=time.time()-start,original_result_sha256=m['original_result_sha256'],
            plan_sha256=m['plan_sha256'],absolute_deadline_unix=m['absolute_deadline_unix'],
            protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=('prepare','run'),required=True);a=p.parse_args()
    (prepare if a.action=='prepare' else run)(a.root)
