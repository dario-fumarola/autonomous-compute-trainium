"""Bounded all16-chip software fixture;no source examples or policy fit."""
import argparse
import json
from pathlib import Path
import signal
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_scale_sharding_fixture import AllChipExecution

IDENTITY='workspace-native-sharded-fixture-v1'
PUBLIC=Path('results/workspace_native_sharded_fixture_20261027')


def prepare(root):
    import os,shutil
    out=root/IDENTITY
    if managed_running():raise RuntimeError('No admission over a Native owner')
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every fixture attempt')
    proof_file=Path('results/workspace_sharded_cpu_20261027/report.json')
    proof=json.loads(proof_file.read_text())
    if proof['status']!='PASS_WORKSPACE_SHARDED_CPU_REFERENCE' or any(file_sha256(Path(n))!=h for n,h in proof['source_sha256'].items()):raise RuntimeError('Current complete CPU sharding proof required')
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available<(512+128)*1024**3:raise RuntimeError('Protected host reserve plus full worker cap required')
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete fixture inode allowance required')
    for p in [root,Path('/')]:
        if shutil.disk_usage(p).free<36*1024**3:raise RuntimeError('Protected output reserve plus complete small fixture allowance')
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()}
    for name in ['scripts/run_native_isolated.sh','tests/test_workspace_sharded_fixture.py','tests/test_workspace_olmo_candidate.py']:
        sources[name]=file_sha256(name)
    worker,host=split_snapshot_sources(sources);start=time.time();names=[IDENTITY+'-'+m for m in ['warm','replay']]
    out.mkdir();PUBLIC.mkdir();plan=dict(identity=IDENTITY,purpose='workspace_native_sharded_software_fixture',registered_unix=start,
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        cpu_proof_sha256=file_sha256(proof_file),world=64,worker_memory_gib=128,seconds_limit=1800,absolute_deadline_unix=start+1800,
        phase_seconds_limit=900,cpu_phase_seconds_limit=60,clock_admission_waiver=True,inode_projection=projection,
        root_comm_ports=reserve_ports(root,names),phases={n:dict(mode=m) for n,m in zip(names,['warm','replay'])},
        actual_numerical_optimizer_updates=4,research_examples=0,research_optimizer_updates=0,policy_fit_admitted=False,
        scope='Numerical arrays only. All64 ranks/all16 chips. Independent unsharded CPU gradients,parameter DELTAS,all AdamW moments,EMA deltas and exact fresh-process model/optimizer/EMA/RNG replay.2% each named-family limit. No larger-model fit is admitted by this fixture.')
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    out=root/IDENTITY;execution=AllChipExecution(root,out/'plan.json',out/'execution');result=dict(status='STOPPED_WORKSPACE_NATIVE_FIXTURE',records={})
    def stop(number,frame):raise KeyboardInterrupt('Owned fixture interrupted')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        for mode in ['warm','replay']:
            name=IDENTITY+'-'+mode
            args=['--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,
                  '--mode',mode,'--comm-port',str(execution.plan['root_comm_ports'][name])]
            if mode=='replay':args+=['--parent','/work/instance-nvme/runs/'+IDENTITY+'-warm']
            result['records'][mode]=execution.native(name,'scripts.qualify_workspace_sharded_fixture',args,'PASS_WORKSPACE_SHARDED_SOFTWARE_FIXTURE')
            write_state(out/'progress.json',result)
        result['status']='PASS_WORKSPACE_NATIVE_SHARDED_SOFTWARE_FIXTURE'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        execution.close(out/'execution-costs.json');result.update(plan_sha256=execution.sha,research_optimizer_updates=0,policy_fit_admitted=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))
        for s,handler in handlers.items():signal.signal(s,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
