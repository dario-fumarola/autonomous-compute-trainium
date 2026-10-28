"""Score already-qualified endpoints after a prelaunch resource-admission stop."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import time
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.workspace_discovery_analysis import analyze
from scripts.run_scale_sharding_fixture import AllChipExecution

IDENTITY='workspace-four-arm-evaluation-recovery-v1'
PUBLIC=Path('results/workspace_discovery_20261027/evaluation-recovery-v1')


def validate_endpoints(root,plan):
    proofs={}
    for arm,count in plan['updates'].items():
        native=root/'runs'/plan['arm_runs'][arm]/'report.json'
        audit=root/('workspace-discovery-audit-'+arm+'-v1/report.json')
        n=json.loads(native.read_text());a=json.loads(audit.read_text())
        if n['status']!='COMPLETE_WORKSPACE_DISCOVERY_ARM_PENDING_MATURE_AUDIT' or n['actual_optimizer_updates']!=count or a['status']!='PASS_WORKSPACE_DISCOVERY_MATURE_CPU_AUDIT':raise AssertionError('Every unchanged completed endpoint and mature audit required')
        proofs[arm]={str(p.relative_to(root)):file_sha256(p) for p in [native,audit]}
    return proofs


def prepare(root):
    if managed_running():raise RuntimeError('Closed training queue required')
    parent=root/'workspace-four-arm-discovery-v3';plan=json.loads((parent/'plan.json').read_text())
    prior=json.loads((parent/'report.json').read_text())
    if prior.get('error')!='Protected host reserve plus complete worker allowance required' or (root/'runs'/(plan['identity']+'-evaluation')).exists():raise AssertionError('Exact zero-score prelaunch admission boundary required')
    proofs=validate_endpoints(root,plan)
    if file_sha256(root/plan['checkpoint_relative'])!=plan['checkpoint_sha256']:raise AssertionError('Common C0 changed')
    if shutil.disk_usage(root).free<36*1024**3:raise RuntimeError('Evaluation artifacts plus35GiB reserve required')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve previous registration')
    sources={str(p):file_sha256(p) for d in ['compute_trainium','scripts','configs'] for p in Path(d).rglob('*.py') if p.is_file()}
    for n in ['scripts/run_native_isolated.sh','docs/WORKSPACE_DISCOVERY_PILOT_20261027.md','tests/test_workspace_evaluation_recovery.py']:
        if Path(n).exists():sources[n]=file_sha256(n)
    worker,host=split_snapshot_sources(sources);projection=project_snapshot_inodes(Path('.'),1)
    import os
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Complete evaluation snapshot inode reserve required')
    phase=IDENTITY+'-evaluation'
    plan.update(identity=IDENTITY,registered_unix=time.time(),seconds_limit=1800,phase_seconds_limit=1800,
        worker_memory_gib=192,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='evaluation')},inode_projection=projection,
        retained_endpoint_proofs=proofs,retained_training_plan_sha256=file_sha256(parent/'plan.json'),
        retained_prior_costs={k:prior[k] for k in ['prior_execution_costs','prior_recovery_costs'] if k in prior},
        retained_training_report_sha256=file_sha256(parent/'report.json'),retained_training_costs=json.loads((parent/'execution-costs.json').read_text()),
        recovery_scope='EVALUATION ONLY after all four full-model audits pass.200GiB admission lacked host headroom;192GiB bounds a no-gradient single actor with no optimizer or EMA.512GiB host/35GiB disk/2% numerical gates unchanged.All checkpoints,data,320-anchor panel,seeds,donors and metrics unchanged.No more fitting,selection,tuning or source exposure.Original overall deadline remains.')
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    out=root/IDENTITY;execution=AllChipExecution(root,out/'plan.json',out/'execution');plan=execution.plan
    result=dict(status='STOPPED_WORKSPACE_DISCOVERY_EVALUATION_RECOVERY',scientific_optimizer_updates=0,
        retained_training_costs=plan['retained_training_costs'],retained_prior_costs=plan.get('retained_prior_costs',{}),confirmation=False)
    def stop(n,f):raise KeyboardInterrupt('Owned evaluation recovery interrupted')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        if validate_endpoints(root,plan)!=plan['retained_endpoint_proofs']:raise AssertionError('Retained endpoints or audits changed')
        phase=IDENTITY+'-evaluation'
        options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--comm-port',str(plan['root_comm_ports'][phase])]
        result['evaluation']=execution.native(phase,'scripts.evaluate_workspace_discovery',options,'COMPLETE_WORKSPACE_DISCOVERY_TARGET_BLIND_EVALUATION')
        result['conditional_analysis']=analyze(root,plan)
        result['status']='COMPLETE_WORKSPACE_DISCOVERY_PILOT'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result['plan_sha256']=execution.sha;execution.close(out/'execution-costs.json')
        for d in [out,PUBLIC]:
            write_state(d/'report.json',result);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
