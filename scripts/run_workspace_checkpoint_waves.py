"""Bounded actual-size writer proof before a controlled scientific replay."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution
IDENTITY='workspace-checkpoint-waves-v3';PUBLIC=Path('results/workspace_checkpoint_waves_20261027/recovery-v3')


def prepare(root):
    if managed_running():raise RuntimeError('Exclusive writer qualification only')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every writer attempt')
    gate=root/'workspace-stable-online-cpu-audit-v1/report.json'
    if json.loads(gate.read_text())['status']!='PASS_STABLE_ONLINE_MATURE_CPU_AUDIT':raise AssertionError('Qualified actual mature state required')
    if shutil.disk_usage(root).free<40*1024**3:raise RuntimeError('Full rank archives plus35GiB protected reserve')
    available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available<(512+200)*1024**3:raise RuntimeError('Complete200GiB worker plus512GiB host reserve required')
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Full snapshot inode allowance')
    old=json.loads((root/'workspace-native-online-throughput-v2/plan.json').read_text())
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()}
    for n in ['scripts/run_native_isolated.sh','tests/test_workspace_sharded_fixture.py']:sources[n]=file_sha256(n)
    worker,host=split_snapshot_sources(sources);start=time.time();phase=IDENTITY+'-native'
    plan=dict(identity=IDENTITY,purpose='workspace_checkpoint_waves_qualification',world=64,worker_memory_gib=200,
        registered_unix=start,seconds_limit=1800,absolute_deadline_unix=start+1800,phase_seconds_limit=1600,cpu_phase_seconds_limit=60,
        clock_admission_waiver=True,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='qualification')},inode_projection=projection,
        checkpoint_relative=old['checkpoint_relative'],model_config=old['model_config'],opening_ids=old['opening_ids'],closing_ids=old['closing_ids'],
        mature_audit_sha256=file_sha256(gate),optimizer_updates=1,scientific_fit_admitted=False,
        scope='Both48-update scientific attempts stop during checkpoint OOM.Restored-state writer test did not reproduce LIVE fused optimizer output representation.This proof takes ONE actual qualification step+EMA before bounded single-rank writes and full perturb/restore of all895tensors.200GiB cap only with full512GiB host reserve;no scientific gate relaxed.')
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    out=root/IDENTITY;execution=AllChipExecution(root,out/'plan.json',out/'execution');result=dict(status='STOPPED_ACTUAL_WORKSPACE_CHECKPOINT_WAVES',planned_qualification_optimizer_updates=1)
    try:
        phase=IDENTITY+'-native'
        result['numerical']=execution.native(phase,'scripts.qualify_workspace_checkpoint_waves',
            ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--comm-port',str(execution.plan['root_comm_ports'][phase])],
            'PASS_ACTUAL_WORKSPACE_CHECKPOINT_WAVES')
        result['status']='PASS_ACTUAL_WORKSPACE_CHECKPOINT_WAVES'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        observed=root/'runs'/(IDENTITY+'-native')/'qualification-updates.json'
        result['observed_qualification_update_receipt']=json.loads(observed.read_text()) if observed.exists() else None
        result['interrupted_update_may_be_unrecorded']=result['status'].startswith('STOPPED')
        execution.close(out/'execution-costs.json');result['plan_sha256']=execution.sha
        for d in [out,PUBLIC]:write_state(d/'report.json',result);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
