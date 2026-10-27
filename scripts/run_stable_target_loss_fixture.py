"""Bounded all16-chip numerical loss diagnostic,zero optimizer updates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution
IDENTITY='workspace-stable-loss-fixture-v1'
PUBLIC=Path('results/workspace_stable_loss_20261027')


def prepare(root):
    if managed_running():raise RuntimeError('Exclusive loss qualification')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Retain fixture history')
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Full snapshot inode reserve')
    for p in [root,Path('/')]:
        if shutil.disk_usage(p).free<36*1024**3:raise RuntimeError('Complete small fixture and protected disk allowance')
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()}
    for n in ['scripts/run_native_isolated.sh','tests/test_stable_target_loss.py']:sources[n]=file_sha256(n)
    worker,host=split_snapshot_sources(sources);start=time.time();phase=IDENTITY+'-native'
    plan=dict(identity=IDENTITY,purpose='stable_target_loss_numerical_fixture',world=64,worker_memory_gib=128,
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,registered_unix=start,
        seconds_limit=600,absolute_deadline_unix=start+600,phase_seconds_limit=500,cpu_phase_seconds_limit=60,
        clock_admission_waiver=True,inode_projection=projection,root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='loss')},
        optimizer_updates=0,research_examples=0,policy_fit_admitted=False,scope='Large-vocabulary numerical logits at several confidence levels.Independent analytic FP64 probability gradients with target derivative minus other-probability sum.Standard versus stable loss,unchanged2% row limit,no CPU fallback.No research examples or model update.')
    out.mkdir();PUBLIC.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    out=root/IDENTITY;execution=AllChipExecution(root,out/'plan.json',out/'execution');result=dict(status='STOPPED_STABLE_LOSS_FIXTURE',policy_fit_admitted=False)
    def stop(n,f):raise KeyboardInterrupt('Owned fixture interrupted')
    old={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        phase=IDENTITY+'-native';result['numerical']=execution.native(phase,'scripts.qualify_stable_target_loss',
            ['--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--comm-port',str(execution.plan['root_comm_ports'][phase])],
            'PASS_STABLE_TARGET_LOSS_NATIVE_FIXTURE')
        result['status']='PASS_STABLE_TARGET_LOSS_NATIVE_FIXTURE'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        execution.close(out/'execution-costs.json');result['plan_sha256']=execution.sha
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))
        for s,h in old.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
