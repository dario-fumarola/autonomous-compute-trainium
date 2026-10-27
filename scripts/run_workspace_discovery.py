"""Frozen four-condition learning diagnostic with qualified, audited endpoints."""
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

IDENTITY='workspace-four-arm-discovery-v1'
PUBLIC=Path('results/workspace_discovery_20261027/main')


def prepare(root):
    from compute_trainium.workspace_discovery import matched_updates,step_matrix_work,donor_indices
    from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig
    from compute_trainium.workspace_view_batches import WorkspaceViewBatches
    if managed_running():raise RuntimeError('Exclusive pilot admission after qualifications')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every registration')
    cpu=Path('results/workspace_discovery_cpu_20261027/report.json');proof=json.loads(cpu.read_text())
    if proof['status']!='PASS_WORKSPACE_DISCOVERY_CPU_HELPERS' or any(file_sha256(n)!=h for n,h in proof['source_sha256'].items()):raise AssertionError('Current loss/control/state/controller CPU proof required')
    gate=root/'workspace-stable-online-cpu-audit-v1/report.json'
    if json.loads(gate.read_text())['status']!='PASS_STABLE_ONLINE_MATURE_CPU_AUDIT':raise AssertionError('Actual fresh-source100warm and independent mature gradient gate required')
    initial=root/'workspace-discovery-initial-ordinary-v1/report.json';reference=json.loads(initial.read_text())
    if reference['status']!='PASS_DISCOVERY_INITIAL_ORDINARY_CPU_REFERENCE':raise AssertionError('New ordinary160-position reference required')
    if file_sha256(root/reference['gradient_relative'])!=reference['gradient_sha256']:raise AssertionError('Independent ordinary reference changed')
    if shutil.disk_usage(root).free<115*1024**3 or shutil.disk_usage('/').free<37*1024**3:raise RuntimeError('Complete endpoint storage allowance required')
    projection=project_snapshot_inodes(Path('.'),5)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('All five complete snapshots/artifacts must fit inode reserve')
    old=json.loads((root/'workspace-native-online-throughput-v2/plan.json').read_text())
    data_root=root/old['data_relative'];data=WorkspaceViewBatches(data_root,'train');tune=WorkspaceViewBatches(data_root,'tune')
    receipt=json.loads((data_root/'report.json').read_text())
    for split in ['train','tune']:
        if any(file_sha256(data_root/n)!=h for n,h in receipt['records'][split]['files'].items()):raise AssertionError('Frozen source arrays changed')
    config=OlmoWorkspaceConfig(**old['model_config']);updates=matched_updates(config)
    if max(updates.values())*64>len(data):raise AssertionError('No epoch wrap or source extension during pilot')
    metadata=[json.loads(line) for line in (root/'workspace-source-views-v2/train-views.jsonl').read_text().splitlines()]
    consumed=max(updates.values())*64;groups={metadata[int(data.order[k])]['group'] for k in range(consumed)}
    audit=next(k for k in range(consumed,len(data)) if metadata[int(data.order[k])]['group'] not in groups)
    tune_meta=[json.loads(line) for line in (root/'workspace-source-views-v2/tune-views.jsonl').read_text().splitlines()]
    eval_groups=[tune_meta[int(tune.order[k])]['group'] for k in range(320)]
    donors=donor_indices(eval_groups,[int(tune.domains[int(tune.order[k])]) for k in range(320)])
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()}
    for n in ['scripts/run_native_isolated.sh','docs/WORKSPACE_DISCOVERY_PILOT_20261027.md','tests/test_workspace_discovery.py','tests/test_workspace_streaming_checkpoint.py']:sources[n]=file_sha256(n)
    worker,host=split_snapshot_sources(sources);start=time.time();phases=[IDENTITY+'-'+arm for arm in ['ordinary','workspace','pause','draft']]+[IDENTITY+'-evaluation']
    plan=dict(identity=IDENTITY,purpose='workspace_four_arm_discovery_pilot',world=64,worker_memory_gib=192,
        registered_unix=start,seconds_limit=18000,absolute_deadline_unix=start+18000,phase_seconds_limit=3600,cpu_phase_seconds_limit=1200,
        clock_admission_waiver=True,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,phases),phases={n:dict(mode=n.rsplit('-',1)[-1]) for n in phases},inode_projection=projection,
        qualification_gate_sha256=file_sha256(gate),ordinary_reference_relative=reference['gradient_relative'],ordinary_reference_sha256=reference['gradient_sha256'],
        cpu_proof_sha256=file_sha256(cpu),
        checkpoint_relative=old['checkpoint_relative'],checkpoint_sha256=file_sha256(root/old['checkpoint_relative']),model_config=old['model_config'],
        opening_ids=old['opening_ids'],closing_ids=old['closing_ids'],data_relative=old['data_relative'],data_report_sha256=file_sha256(data_root/'report.json'),
        updates=updates,nominal_matrix_work_per_local_update={arm:step_matrix_work(config,arm) for arm in updates},
        lr=1e-5,schedule='constant_short_learning_diagnostic_no_convergence_claim',betas=[.9,.95],weight_decay=.1,gradient_clip=1.,gradient_scale=256,
        sampling_seed_base=7351,group_size=4,workspace_length=8,reward_horizon=4,ema_decay=.99,ordinary_weight=.5,reader_weight=.5,policy_weight=.1,draft_weight=.1,
        pause_token_id=100257,mature_audit_index=audit,evaluation_anchors=320,evaluation_seed=8123,evaluation_donors=donors,evaluation_groups=eval_groups,
        tuning_opportunities_per_arm=1,terminal_alpha_spent=0,full_from_scratch_admitted=False,
        scope=Path('docs/WORKSPACE_DISCOVERY_PILOT_20261027.md').read_text())
    if plan['checkpoint_sha256']!='883b05baa6f104f9ecabd2a2d1e5b986dc99ee80f22e350312591b46e9828db2':raise AssertionError('Exact common C0 identity required')
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    out=root/IDENTITY;execution=AllChipExecution(root,out/'plan.json',out/'execution');plan=execution.plan
    result=dict(status='STOPPED_WORKSPACE_DISCOVERY_PILOT',conditions={},full_from_scratch_admitted=False)
    def stop(n,f):raise KeyboardInterrupt('Owned fixed-budget discovery interrupted')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    def options(phase):return ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--comm-port',str(plan['root_comm_ports'][phase])]
    try:
        for arm in ['ordinary','workspace','pause','draft']:
            # Reserve the WHOLE uncompressed next checkpoint + gradient even
            # though the actual archives use exact lossless compression.
            needed=66 if arm=='workspace' else 60
            if shutil.disk_usage(root).free<needed*1024**3:raise RuntimeError('Complete next endpoint plus35GiB protected reserve required')
            phase=IDENTITY+'-'+arm
            native=execution.native(phase,'scripts.train_workspace_discovery',options(phase)+['--arm',arm],
                'COMPLETE_WORKSPACE_DISCOVERY_ARM_PENDING_MATURE_AUDIT')
            result['conditions'][arm]=dict(native=native,audit_pending=True);write_state(PUBLIC/'progress.json',result)
            audit_id='workspace-discovery-audit-'+arm+'-v1'
            audit=execution.cpu(audit_id,'scripts.audit_workspace_discovery',
                ['--root',str(root),'--plan',str(out/'plan.json'),'--arm',arm],root/audit_id/'report.json','PASS_WORKSPACE_DISCOVERY_MATURE_CPU_AUDIT')
            result['conditions'][arm].update(audit=audit,audit_pending=False);write_state(PUBLIC/'progress.json',result)
        phase=IDENTITY+'-evaluation'
        result['evaluation']=execution.native(phase,'scripts.evaluate_workspace_discovery',options(phase),'COMPLETE_WORKSPACE_DISCOVERY_TARGET_BLIND_EVALUATION')
        from compute_trainium.workspace_discovery_analysis import analyze
        result['conditional_analysis']=analyze(root,plan)
        result['status']='COMPLETE_WORKSPACE_DISCOVERY_PILOT'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        observed={}
        for arm in ['ordinary','workspace','pause','draft']:
            path=root/'runs'/(IDENTITY+'-'+arm)/'updates.json'
            if path.exists():
                try:observed[arm]=len(json.loads(path.read_text()))
                except (OSError,ValueError):observed[arm]='receipt_incomplete'
        result['observed_completed_optimizer_updates']=observed
        result['interrupted_update_may_be_unrecorded']=result['status'].startswith('STOPPED')
        result['plan_sha256']=execution.sha;execution.close(out/'execution-costs.json')
        for d in [out,PUBLIC]:
            write_state(d/'report.json',result);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
