"""A single fixed2x2 zero-update follow-up to the complete negative pilot."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import time
import os
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.workspace_readout_diagnostic import analyze_readouts
from scripts.resume_workspace_discovery_evaluation import validate_endpoints
from scripts.run_scale_sharding_fixture import AllChipExecution

IDENTITY='workspace-readout-diagnostic-v1'
PUBLIC=Path('results/workspace_readouts_20261028')


def prepare(root):
    if managed_running():raise RuntimeError('Previous evaluator must be closed')
    parent=root/'workspace-four-arm-evaluation-recovery-v1';original=json.loads((parent/'report.json').read_text())
    if original['status']!='COMPLETE_WORKSPACE_DISCOVERY_PILOT':raise AssertionError('Original complete negative remains authoritative')
    plan=json.loads((parent/'plan.json').read_text());proofs=validate_endpoints(root,plan)
    if proofs!=plan['retained_endpoint_proofs']:raise AssertionError('No altered fitted endpoint')
    if file_sha256(root/plan['checkpoint_relative'])!=plan['checkpoint_sha256']:raise AssertionError('C0 changed')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('No overwrite or silent extra readout search')
    sources={str(p):file_sha256(p) for d in ['compute_trainium','scripts','configs'] for p in Path(d).rglob('*.py') if p.is_file()}
    for n in ['scripts/run_native_isolated.sh','docs/WORKSPACE_READOUT_DIAGNOSTIC_20261028.md','tests/test_workspace_readouts.py','tests/test_workspace_native_objective.py']:sources[n]=file_sha256(n)
    worker,host=split_snapshot_sources(sources);projection=project_snapshot_inodes(Path('.'),1)
    if os.statvfs(root).f_favail<projection['required_free_inodes'] or shutil.disk_usage(root).free<36*1024**3:raise RuntimeError('Complete snapshot and artifact reserve required')
    phase=IDENTITY+'-evaluation';plan.update(identity=IDENTITY,purpose='workspace_readout_diagnostic',registered_unix=time.time(),seconds_limit=1800,phase_seconds_limit=1800,
        worker_memory_gib=192,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='evaluation')},inode_projection=projection,
        original_evaluation_run=plan['identity']+'-evaluation',original_report_sha256=file_sha256(parent/'report.json'),
        diagnostic_scope=Path('docs/WORKSPACE_READOUT_DIAGNOSTIC_20261028.md').read_text(),scientific_optimizer_updates=0,terminal_alpha_spent=0)
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    out=root/IDENTITY;e=AllChipExecution(root,out/'plan.json',out/'execution');p=e.plan
    result=dict(status='STOPPED_WORKSPACE_READOUT_DIAGNOSTIC',scientific_optimizer_updates=0,confirmation=False)
    def stop(n,f):raise KeyboardInterrupt('Owned bounded readout interrupted')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        if validate_endpoints(root,p)!=p['retained_endpoint_proofs']:raise AssertionError('Endpoint proofs changed')
        phase=IDENTITY+'-evaluation';options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase])]
        result['evaluation']=e.native(phase,'scripts.evaluate_workspace_readouts',options,'COMPLETE_WORKSPACE_READOUT_DIAGNOSTIC')
        result['analysis']=analyze_readouts(root,p);result['status']='COMPLETE_WORKSPACE_READOUT_DIAGNOSTIC'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result['plan_sha256']=e.sha;e.close(out/'execution-costs.json')
        for d in [out,PUBLIC]:write_state(d/'report.json',result);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');p.add_argument('--recovery',action='store_true');a=p.parse_args()
    if a.recovery:IDENTITY='workspace-readout-diagnostic-v2';PUBLIC=Path('results/workspace_readouts_20261028/recovery-v2')
    run(a.root) if a.run else prepare(a.root)
