"""Bounded all-chip whole-online-cost qualification,not a scientific fit."""
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

IDENTITY='workspace-native-online-throughput-v1'
PUBLIC=Path('results/workspace_online_throughput_20261027')


def prepare(root):
    if managed_running():raise RuntimeError('Exclusive online qualification only')
    cpu_proof=Path('results/workspace_online_cpu_20261027/report.json')
    proof=json.loads(cpu_proof.read_text())
    if proof['status']!='PASS_WORKSPACE_ONLINE_CPU_HELPERS' or any(file_sha256(n)!=h for n,h in proof['source_sha256'].items()):raise AssertionError('Current complete CPU online/state helper proof required')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Retain every online attempt')
    gate=root/'workspace-full-update-cpu-audit-v1/report.json'
    if json.loads(gate.read_text())['status']!='PASS_WORKSPACE_SAVED_UPDATE_CPU_AUDIT':raise AssertionError('Independent complete saved-state CPU update audit required')
    parent=root/'workspace-native-full-update-v1/plan.json';original=json.loads(parent.read_text())
    if json.loads((parent.parent/'report.json').read_text())['status']!='PASS_WORKSPACE_FULL_MODEL_UPDATE_AND_RESTORE':raise AssertionError('Complete actual-size Native update/recovery gate required')
    for path,minimum in [(root,66),(Path('/'),37)]:
        if shutil.disk_usage(path).free<minimum*1024**3:raise RuntimeError('Full final checkpoint/gradient plus protected filesystem reserve required')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<(512+192)*1024**3:raise RuntimeError('Full192GiB worker plus512GiB host reserve required')
    projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Full snapshot/artifact inode allowance required')
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()}
    for n in ['scripts/run_native_isolated.sh','tests/test_workspace_native_objective.py','tests/test_workspace_sharded_fixture.py']:sources[n]=file_sha256(n)
    worker,host=split_snapshot_sources(sources);start=time.time();phase=IDENTITY+'-online'
    shards='runs/workspace-native-full-update-v1-gradient'
    plan=dict(identity=IDENTITY,purpose='workspace_online_throughput_qualification',world=64,worker_memory_gib=192,
        registered_unix=start,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        seconds_limit=5400,absolute_deadline_unix=start+5400,phase_seconds_limit=5100,cpu_phase_seconds_limit=60,
        clock_admission_waiver=True,inode_projection=projection,root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='online')},
        parent_shards_relative=shards,parent_shard_sha256=[file_sha256(root/shards/f'rank{i}-u1.zip') for i in range(64)],
        parent_plan_sha256=file_sha256(parent),saved_state_cpu_audit_sha256=file_sha256(gate),
        checkpoint_relative=original['checkpoint_relative'],model_config=original['model_config'],
        inputs_relative=original['inputs_relative'],inputs_sha256=original['inputs_sha256'],opening_ids=original['opening_ids'],closing_ids=original['closing_ids'],
        cpu_proof_sha256=file_sha256(cpu_proof),post_restore_sampling_seed_base=7351,lr=1e-5,betas=[.9,.95],epsilon=1e-8,weight_decay=.1,gradient_clip=1.,gradient_scale=256,ema_decay=.99,ordinary_weight=.5,reader_weight=.5,policy_weight=.1,cost_per_token=.001,cold_updates=1,warm_updates=100,actual_qualification_policy_optimizer_updates=101,scientific_policy_fit_updates=0,
        group_size=4,workspace_length=8,reward_horizon=4,padded_positions=192,global_source_groups=1,
        policy_fit_admitted=False,scope='Fresh-process exact full actor/scorer/AdamW/RNG restore from saved u1,then one cold+100warm online generate/score/gradient/AdamW/EMA updates on ONE repeated accepted TRAIN anchor. Independent per-rank exploration,post-update EMA .99,no teacher. Whole costs and all actual exposures logged. Final102-counter state and mature zero-update full gradient saved for independent CPU audit. No scientific fit,selection on generated thought quality,confirmation or original from-scratch claim.')
    if file_sha256(root/plan['inputs_relative'])!=plan['inputs_sha256']:raise AssertionError('Qualified source inputs changed')
    out.mkdir();PUBLIC.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    out=root/IDENTITY;execution=AllChipExecution(root,out/'plan.json',out/'execution');result=dict(status='STOPPED_WORKSPACE_ONLINE_THROUGHPUT',policy_fit_admitted=False)
    def stop(n,f):raise KeyboardInterrupt('Owned fixed-update qualification interrupted')
    old={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    phase=IDENTITY+'-online'
    try:
        result['numerical']=execution.native(phase,'scripts.qualify_workspace_online_throughput',
            ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,
             '--comm-port',str(execution.plan['root_comm_ports'][phase])],
            'PASS_WORKSPACE_ONLINE_THROUGHPUT_EXECUTION_PENDING_MATURE_AUDIT')
        result['status']='COMPLETE_ONLINE_THROUGHPUT_PENDING_MATURE_CPU_AUDIT'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        updates_file=root/'runs'/phase/'updates.json'
        result['observed_completed_qualification_optimizer_updates']=len(json.loads(updates_file.read_text())) if updates_file.exists() else 0
        result['interrupted_update_may_be_unrecorded']=result['status'].startswith('STOPPED')
        execution.close(out/'execution-costs.json');result['plan_sha256']=execution.sha
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))
        for s,h in old.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
