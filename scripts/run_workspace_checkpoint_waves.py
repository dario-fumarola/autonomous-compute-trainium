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
IDENTITY='workspace-checkpoint-waves-v1';PUBLIC=Path('results/workspace_checkpoint_waves_20261027')


def prepare(root):
    if managed_running():raise RuntimeError('Exclusive writer qualification only')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every writer attempt')
    gate=root/'workspace-stable-online-cpu-audit-v1/report.json'
    if json.loads(gate.read_text())['status']!='PASS_STABLE_ONLINE_MATURE_CPU_AUDIT':raise AssertionError('Qualified actual mature state required')
    if shutil.disk_usage(root).free<40*1024**3:raise RuntimeError('Eight compressed rank archives plus35GiB protected reserve')
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Full snapshot inode allowance')
    old=json.loads((root/'workspace-native-online-throughput-v2/plan.json').read_text())
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()}
    for n in ['scripts/run_native_isolated.sh','tests/test_workspace_sharded_fixture.py']:sources[n]=file_sha256(n)
    worker,host=split_snapshot_sources(sources);start=time.time();phase=IDENTITY+'-native'
    plan=dict(identity=IDENTITY,purpose='workspace_checkpoint_waves_qualification',world=64,worker_memory_gib=192,
        registered_unix=start,seconds_limit=1800,absolute_deadline_unix=start+1800,phase_seconds_limit=1600,cpu_phase_seconds_limit=60,
        clock_admission_waiver=True,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='qualification')},inode_projection=projection,
        checkpoint_relative=old['checkpoint_relative'],model_config=old['model_config'],opening_ids=old['opening_ids'],closing_ids=old['closing_ids'],
        mature_audit_sha256=file_sha256(gate),optimizer_updates=0,scientific_fit_admitted=False,
        scope='Preserved v1 stopped after48workspace updates during checkpoint OOM.New zero-update actual-size qualified-state proof:runtime scalar inputs,released compiled wrappers,eight concurrent full-state writers,unchanged named2% gradient threshold.')
    out.mkdir();PUBLIC.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    out=root/IDENTITY;execution=AllChipExecution(root,out/'plan.json',out/'execution');result=dict(status='STOPPED_ACTUAL_WORKSPACE_CHECKPOINT_WAVES',optimizer_updates=0)
    try:
        phase=IDENTITY+'-native'
        result['numerical']=execution.native(phase,'scripts.qualify_workspace_checkpoint_waves',
            ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--comm-port',str(execution.plan['root_comm_ports'][phase])],
            'PASS_ACTUAL_WORKSPACE_CHECKPOINT_WAVES')
        result['status']='PASS_ACTUAL_WORKSPACE_CHECKPOINT_WAVES'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        execution.close(out/'execution-costs.json');result['plan_sha256']=execution.sha
        for d in [out,PUBLIC]:write_state(d/'report.json',result);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
