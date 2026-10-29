"""Finish only missing mature/state checks; never a third full warm-up."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution

ID='chunked-persistent-completion-v1'
PUBLIC=Path('results/chunked_persistent_workspace_20261029/completion-v1')


def check_partial_evidence(checks,updates):
    if set(checks)!={'0','1','2','3'}:raise AssertionError('All four actual-shape cold checks required')
    for c in checks.values():
        if abs(c['loss_error'])>.003 or max(v['relative_l2'] for v in c.values() if isinstance(v,dict))>.02:
            raise AssertionError('Numerical failure cannot be bypassed')
    warm=[r for r in updates if 4<=r['update']<=103]
    if len(warm)!=100 or {r['update'] for r in warm}!=set(range(4,104)):
        raise AssertionError('Original complete100 warm updates required')
    times={str(k):sum(r['seconds'] for r in warm if r['extra_steps']==k)/sum(r['extra_steps']==k for r in warm) for k in range(4)}
    if max(times.values())>1 or times['2']/times['0']>3:
        raise AssertionError('Original throughput gates remain binding')
    return times


def prepare(root,parent_choice,finish):
    import torch
    from scripts.qualify_native import digest
    if managed_running():raise RuntimeError('Previous owners must be closed')
    prior=root/'chunked-persistent-qualification-recovery-v1/report.json'
    failure=json.loads(prior.read_text())
    if failure['status']!='STOPPED_CHUNKED_PERSISTENT_QUALIFICATION' or failure.get('error_type')!='TimeoutError':
        raise AssertionError('Only the diagnosed bounded timeout permits this completion')
    if shutil.disk_usage(root).free<39*1024**3 or shutil.disk_usage('/').free<35*1024**3:
        raise RuntimeError('Four GiB completion allowance and protected reserves')
    parentid={'initial':'chunked-persistent-qualification-v1','recovery':'chunked-persistent-qualification-recovery-v1'}[parent_choice]
    parent=root/parentid;old=json.loads((parent/'plan.json').read_text());run=root/'runs'/old['warm_phase']
    for f in ('compute_trainium/chunked_persistent_workspace.py','compute_trainium/chunked_persistent_factory.py',
        'compute_trainium/depth_causal_workspace.py','compute_trainium/tied_interleaved_ordinary.py',
        'compute_trainium/model.py','compute_trainium/interleaved_objective.py','scripts/qualify_interleaved_grouped.py'):
        if file_sha256(f)!=old['source_sha256'][f]:raise AssertionError('Qualified mathematical execution changed')
    checks=json.loads((run/'cold-local-checks.json').read_text());updates=json.loads((run/'updates.json').read_text())
    times=check_partial_evidence(checks,updates)
    checkpoint=run/'PRIVATE-u103.pt';checkpoint_hash=file_sha256(checkpoint)
    required=[103] if finish else [103,104]
    states={}
    for u in required:
        f=run/f'PRIVATE-u{u}.pt';saved=torch.load(f,weights_only=False,map_location='cpu',mmap=True)
        if saved['update']!=u or digest({k:saved[k] for k in ('model','optimizer','rng')})!=saved['digest']:
            raise AssertionError('Complete preserved full-state checkpoint required')
        states[str(u)]=dict(file_sha256=file_sha256(f),state_digest=saved['digest']);del saved
    if not finish:
        gradient=run/'PRIVATE-mature-gradient.pt';g=torch.load(gradient,weights_only=True,map_location='cpu',mmap=True)
        if g['update']!=104 or g['k']!=2:raise AssertionError('Complete correct mature gradient required')
        states['gradient']=dict(file_sha256=file_sha256(gradient));del g
    data=root/old['data_relative'];dr=json.loads((data/'report.json').read_text())
    if file_sha256(data/'report.json')!=old['data_report_sha256'] or any(file_sha256(data/f)!=h for f,h in dr['files'].items()):
        raise AssertionError('Pinned disposable TRAIN data changed')
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/CHUNKED_PERSISTENT_COMPLETION_20261029.md',
              'tests/test_chunked_persistent_completion.py'):
        files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);phases=([ID+'-finish'] if finish else [])+[ID+'-replay']
    projection=project_snapshot_inodes(Path('.'),len(phases))
    if os.statvfs(root).f_favail<projection['required_free_inodes']:
        raise RuntimeError('All remaining snapshots and inode reserve required')
    now=time.time();p=copy.deepcopy(old)
    p.update(identity=ID,registered_unix=now,absolute_deadline_unix=now+2700,seconds_limit=2700,
        phase_seconds_limit=1200,cpu_phase_seconds_limit=900,
        source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,phases),phases={n:{} for n in phases},inode_projection=projection,
        completion_checkpoint_relative=str(checkpoint.relative_to(root)),completion_checkpoint_sha256=checkpoint_hash,
        preserved_full_states=states,parent_plan_relative=str((parent/'plan.json').relative_to(root)),
        parent_plan_sha256=file_sha256(parent/'plan.json'),prior_failure_sha256=file_sha256(prior),
        old_qualification_deadline_unix=old['absolute_deadline_unix'],finish_update_needed=finish,
        warm_phase=ID+'-finish' if finish else old['warm_phase'],original_warm_times=times,
        cold_check_sha256=file_sha256(run/'cold-local-checks.json'),warm_trace_sha256=file_sha256(run/'updates.json'),
        artifact_allowance_bytes=4*1024**3,
        scope='Supplemental validation of preserved disposable states after the original qualification budget failed. Reuse only completed cold checks and100 warm timings; perform missing generated-state update if required, fresh-process full-state replay and independent full-global CPU gradient audit. No third full warm-up, threshold relaxation, scientific fit or weight promotion. Original attempts and deadlines remain failed and their costs stay charged.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution')
    result=dict(status='RUNNING_CHUNKED_PERSISTENT_COMPLETION',records={})
    try:
        if e.plan['finish_update_needed']:
            name=ID+'-finish'
            result['records']['finish']=e.native(name,'scripts.complete_chunked_persistent_checkpoint',
                ['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
                 '--plan-sha',e.sha,'--comm-port',str(e.plan['root_comm_ports'][name])],
                'COMPLETE_CHUNKED_CHECKPOINT_UPDATE_PENDING_AUDITS')
            write_state(PUBLIC/'progress.json',result)
        name=ID+'-replay'
        result['records']['replay']=e.native(name,'scripts.chunked_persistent_entry',
            ['--entry','qualification','--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
             '--plan-sha',e.sha,'--mode','replay','--parent',f"/work/instance-nvme/runs/{e.plan['warm_phase']}",
             '--comm-port',str(e.plan['root_comm_ports'][name])], 'PASS_INTERLEAVED_WARM_EXACT_REPLAY')
        write_state(PUBLIC/'progress.json',result);dest=out/'cpu-audit'
        result['records']['audit']=e.cpu(ID+'-audit','scripts.chunked_persistent_entry',
            ['--entry','audit','--root',str(root),'--plan',str(out/'plan.json'),
             '--plan-sha',e.sha,'--output',str(dest)],dest/'report.json','PASS_INTERLEAVED_MATURE_GLOBAL_CPU')
        result.update(status='PASS_CHUNKED_PERSISTENT_COMPLETION',original_warm_times=e.plan['original_warm_times'],
            scientific_fit_admitted=False,retained_for_science=False,scope=e.plan['scope'])
    except BaseException as error:
        result.update(status='STOPPED_CHUNKED_PERSISTENT_COMPLETION',error_type=type(error).__name__,error=str(error));raise
    finally:
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        e.close(PUBLIC/'execution-costs.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run'],required=True)
    p.add_argument('--parent',choices=['initial','recovery'],default='recovery');p.add_argument('--finish-update',action='store_true')
    a=p.parse_args();prepare(a.root,a.parent,a.finish_update) if a.action=='prepare' else run(a.root)
