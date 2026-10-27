"""Preparation and separately activated numerical-only head-free fixed-two gate."""
import argparse
import concurrent.futures as cf
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np

from compute_trainium.execution_isolation import managed_running
from compute_trainium.fixed_two_curve import stop_on_failed_future
from compute_trainium.head_free_fixed_two_gate import STATUS,TRIALS,validate_report
from compute_trainium.head_free_fixed_two_v2_gate import parent_binding,artifact_budget,FIXTURE_STATUS,fixture_gate
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_fixed_two_ntp_tune import GuardedExecution
from scripts.run_fixed_control_replication import reserves

IDENTITY='head-free-fixed2-numerical-v2-recovery'
FAILED='head-free-fixed2-numerical-v2'
PARENT='late-two-allocation-main-v1'
PUBLIC=Path('results/head_free_fixed2_numerical_v2_20261027/recovery')
DOC='docs/HEAD_FREE_FIXED_TWO_PHASE_RECOVERY_20261027.md'


def read(file):return json.loads(Path(file).read_text())


def all_sources(parent_sources):
    result={file:file_sha256(file) for file in parent_sources}
    for tree in ['compute_trainium','scripts','configs','third_party']:
        for file in Path(tree).rglob('*.py'):
            if file.is_file() and not file.is_symlink():result[str(file)]=file_sha256(file)
    for file in [DOC,'tests/test_head_free_fixed_two_gate.py','tests/test_head_free_fixed_two_v2_gate.py',
            'tests/test_rejected_replica_evidence.py','tests/test_execution_failure_preservation.py',
            'tests/test_head_free_fixed_two_phase_recovery.py',
            'docs/REJECTED_REPLICA_EVIDENCE_20261027.md','scripts/run_native_isolated.sh']:
        result[file]=file_sha256(file)
    return result


def phase_name(trial,mode):
    # Logical trial keys retain the original proof schema; worker IDs are safe.
    name=IDENTITY+'-'+trial.replace('_','-')+'-'+mode
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name):
        raise ValueError('Safe registered Native phase identity required')
    return name


def prepare(root):
    """CPU metadata only;never activates or occupies a native worker."""
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every numerical registration')
    closure=parent_binding(root)
    prior_file=root/FAILED/'report.json';prior=read(prior_file)
    prior_plan=root/FAILED/'plan.json';activation=read(root/FAILED/'activation.json')
    if (prior['status']!='STOPPED_HEAD_FREE_FIXED_TWO_NUMERICAL_QUALIFICATION'
            or prior['records'] or prior['error']!='Explicit safe phase identity and half required'
            or prior['plan_sha256']!=file_sha256(prior_plan)):
        raise AssertionError('Exact stopped zero-model-update numerical predecessor required')
    original_deadline=activation['activated_unix']+read(prior_plan)['seconds_limit']
    reserves(root)
    parity_file=root/'head-free-recurrent-cpu-parity-v1/report.json';parity=read(parity_file)
    if (parity['status']!='PASS_HEAD_FREE_RECURRENT_CPU_PREDICTION_AND_GRADIENT_PARITY'
            or parity['model_forwards']!=8 or parity['optimizer_updates']!=0
            or len(parity['records'])!=4 or not all(r['all_body_gradients_bitwise_equal'] for r in parity['records'])):
        raise ValueError('Complete original/mature CPU parity required')
    old_file=root/'lean-ntp-native-preparation-v1/plan.json';old=read(old_file)
    reference=root/'parcae-low-depth-qualification-v1';reference_plan=read(reference/'plan.json')
    parent_file=root/PARENT/'plan.json';parent=read(parent_file)
    if old['model']!=parent['model'] or reference_plan['model']!=old['model']:
        raise ValueError('Original qualified family geometry must remain unchanged')
    selected=[c for c in old['cases'] if c['arm']=='poisson2' and c['accumulation']==16]
    if len(selected)!=1:raise ValueError('Exactly one full-two numerical geometry required')
    table_file=old_file.parent/'poisson2-a16-depths.npy';table=np.load(table_file,mmap_mode='r')
    if (file_sha256(table_file)!=selected[0]['depth_table_sha256']
            or table.shape!=(121,16,32,2) or not np.all(table==2)):
        raise ValueError('Actual numeric full-two/full-backward table required')
    initial=old['initial_states']['family'];initial_file=(old_file.parent/initial['file']).resolve()
    if file_sha256(initial_file)!=initial['sha256']:raise AssertionError('Original unupdated qualification initializer changed')
    prior=read(root/'baseline-numerical-recovery-v1/report.json')['records']['one']
    tune=read(root/'late-two-allocation-tune-v1/report.json')
    step=max(v['two_pass_seconds']/100 for v in tune['actual_warm'].values())
    audit=max(900.,prior['independent_cpu']['wall_seconds']*1.5)
    replay=max(120.,prior['restored']['wall_seconds']*1.5)
    parts=dict(two_parallel_warm_replay_training=122*step*1.5,cold_native_phases=4*250/2,
        parallel_independent_audits=audit,parallel_replay_audits=replay,zero_update_native_fixture=300.,closure_margin=300.)
    estimate=sum(parts.values())
    if estimate+120>3600 or time.time()+estimate+120>original_deadline:raise RuntimeError('Complete numerical-only estimate exceeds frozen cap')
    warm_dir=root/'runs/baseline-numerical-v1-one-warm'
    replay_dir=root/'runs/baseline-numerical-recovery-v1-one-replay'
    warm_reference=read(warm_dir/'report.json')
    if (warm_reference['initial_model_digest']!=initial['model_digest']
            or warm_reference['signature']['precision']!='fp16'):
        raise AssertionError('Matching head-free family/precision artifact reference required')
    artifacts=artifact_budget(sum(p.stat().st_size for p in warm_dir.rglob('*') if p.is_file()),
        sum(p.stat().st_size for p in replay_dir.rglob('*') if p.is_file()),141963840)
    if shutil.disk_usage(root).free<artifacts['required_free_nvme_bytes']:
        raise RuntimeError('Complete normal and both-group rejected-state artifacts do not fit')
    for filesystem in [Path('/'),root.parent]:
        if shutil.disk_usage(filesystem).free<artifacts['required_free_other_filesystem_bytes']:
            raise RuntimeError('Other mounted filesystem protected reserve plus1GiB allowance required')
    sources=all_sources(parent['source_sha256']);worker,host=split_snapshot_sources(sources)
    changed=[f for f,h in parent['source_sha256'].items() if sources[f]!=h]
    if set(changed)!={'compute_trainium/lean_baseline_science.py','scripts/qualify_baseline_managed.py',
            'scripts/audit_baseline_native.py','scripts/train_lean_baseline.py',
            'scripts/train_future_span_scientific.py','scripts/train_low_depth_scientific.py',
            'scripts/train_parcae_precise_qualification.py','compute_trainium/future_span_execution.py'}:
        raise AssertionError('Only role wrappers,checkpoint diagnostics and bounded host execution may differ;model/optimizer kernels unchanged')
    out.mkdir();inputs=out/'inputs';inputs.mkdir()
    prep=copy.deepcopy(old);prep.update(identity=IDENTITY+'-inputs',source_sha256=sources,
        worker_source_sha256=worker,host_only_source_sha256=host,cases=selected,
        qualification_only=True,scientific_main_admitted=False,native_launch_allowed=False,
        scope=Path(DOC).read_text())
    prep['initial_states']={'family':copy.deepcopy(initial)}
    prep['initial_states']['family']['file']=os.path.relpath(initial_file,inputs)
    shutil.copyfile(table_file,inputs/'poisson2-a16-depths.npy')
    write_state(inputs/'plan.json',prep)
    write_state(inputs/'cpu-report.json',dict(status='COMPLETE_PARCAE_TRAINING_QUALIFICATION_PREPARATION',
        model_updates=0,assigned_chips=0,scope='Original unupdated qualification C0 and byte-identical deterministic full-two/full-backward table. CPU parity complete;Native remains unqualified.'))
    profile=dict(preparation_directory=str(inputs.relative_to(root)),preparation_plan_sha256=file_sha256(inputs/'plan.json'),
        arm='poisson2',parameters=141963840,initial_model_digest=initial['model_digest'],
        reference_preparation_directory=str(reference.relative_to(root)),reference_plan_sha256=file_sha256(reference/'plan.json'),reference_root_view='.')
    phases={}
    for trial in sorted(TRIALS):
        for mode in ['warm','replay']:
            name=phase_name(trial,mode)
            phases[name]=dict(profile='fixed2',trial=trial,mode=mode,parent=phase_name(trial,'warm') if mode=='replay' else None)
    fixture_phase=IDENTITY+'-rejected-state-fixture'
    phases[fixture_phase]=dict(profile='rejected_state_fixture',mode='zero_update',parent=None)
    plan=dict(identity=IDENTITY,registered_unix=time.time(),purpose='baseline_native_qualification_only',
        model=old['model'],parameters=141963840,gradient_rule='full_fixed_two',depth_table_sha256=file_sha256(table_file),
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        profiles={'fixed2':profile},phases=phases,root_comm_ports=reserve_ports(root,list(phases)),
        seconds_limit=3600,phase_seconds_limit=900,cpu_phase_seconds_limit=1800,
        measured_budget=dict(parts=parts,complete_seconds=estimate,cap_seconds=3600),
        cpu_parity_report_sha256=file_sha256(parity_file),parent_main_plan_sha256=file_sha256(parent_file),
        parent_must_be_complete_qualified=False,parent_closure_binding=closure,
        original_unexecuted_v1_plan_sha256=file_sha256(root/'head-free-fixed2-numerical-v1/plan.json'),
        rejected_state_evidence_required=True,fixture_phase=fixture_phase,
        native_phase_seconds_limits={fixture_phase:300},artifact_budget=artifacts,
        artifact_reference_reports_sha256={str((d/'report.json').relative_to(root)):file_sha256(d/'report.json') for d in [warm_dir,replay_dir]},
        original_absolute_deadline_unix=original_deadline,
        failed_numeric_plan_sha256=file_sha256(prior_plan),failed_numeric_report_sha256=file_sha256(prior_file),
        activation_required=True,scientific_fit_admitted=False,
        inode_projection=project_snapshot_inodes(Path.cwd(),5),scope=Path(DOC).read_text(),
        prospective_role_wrapper_changes_from_main=changed,
        actual_qualification_updates_planned=244,research_optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0)
    PUBLIC.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    write_state(PUBLIC/'status.json',dict(status='PREPARED_NUMERICAL_ONLY_NOT_ACTIVATED',assigned_chips=0,scientific_fit_admitted=False))
    return plan


def activate(root,plan_file,plan):
    if managed_running():raise RuntimeError('All predecessor native owners must be closed')
    if (file_sha256(root/FAILED/'plan.json')!=plan['failed_numeric_plan_sha256']
            or file_sha256(root/FAILED/'report.json')!=plan['failed_numeric_report_sha256']
            or time.time()+plan['measured_budget']['complete_seconds']+120>plan['original_absolute_deadline_unix']):
        raise RuntimeError('Exact predecessor and complete remaining budget within original clock required')
    closure=parent_binding(root)
    if closure!=plan['parent_closure_binding']:
        raise ValueError('Exact stopped-pair and audited-single closure changed')
    original=root/'head-free-fixed2-numerical-v1'
    if (file_sha256(original/'plan.json')!=plan['original_unexecuted_v1_plan_sha256']
            or (original/'activation.json').exists()):
        raise AssertionError('Original unexecuted registration must remain preserved and inactive')
    if file_sha256(root/'head-free-recurrent-cpu-parity-v1/report.json')!=plan['cpu_parity_report_sha256']:
        raise AssertionError('Bound CPU parity report changed')
    if any(file_sha256(f)!=h for f,h in plan['source_sha256'].items()):raise AssertionError('Prepared numerical source changed')
    reserves(root)
    if shutil.disk_usage(root).free<plan['artifact_budget']['required_free_nvme_bytes']:raise RuntimeError('Complete numerical artifacts plus original filesystem reserve required')
    if os.statvfs(root).f_favail<plan['inode_projection']['required_free_inodes']:raise RuntimeError('All five snapshots plus original inode reserve required')
    for filesystem in [Path('/'),root.parent]:
        if shutil.disk_usage(filesystem).free<plan['artifact_budget']['required_free_other_filesystem_bytes']:
            raise RuntimeError('Complete other-filesystem allowance does not fit')
    out=plan_file.parent
    if (out/'activation.json').exists():raise FileExistsError('Preserve original activation clock;explicit recovery required')
    activation=dict(activated_unix=time.time(),closed_parent_binding=closure,
        plan_sha256=file_sha256(plan_file),purpose='Numerical qualification only;no scientific fit admitted')
    write_state(out/'activation.json',activation);write_state(PUBLIC/'activation.json',activation)
    return activation


class CheckedExecution(GuardedExecution):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        activation=read(self.root/IDENTITY/'activation.json')
        self.deadline=min(self.deadline,activation['activated_unix']+self.plan['seconds_limit'],self.plan['original_absolute_deadline_unix'])
        self.execution.update(deadline_unix=self.deadline,original_activated_unix=activation['activated_unix'],
            original_predecessor_deadline_unix=self.plan['original_absolute_deadline_unix'])
        write_state(self.out/'execution-start.json',self.execution)
    def check(self):
        super().check();reserves(self.root)
        activation=read(self.root/IDENTITY/'activation.json')
        if parent_binding(self.root)!=activation['closed_parent_binding']:
            raise AssertionError('Preserved closed-parent chain changed after activation')
        for profile in self.plan['profiles'].values():
            for path,key in [('preparation_directory','preparation_plan_sha256'),('reference_preparation_directory','reference_plan_sha256')]:
                if file_sha256(self.root/profile[path]/'plan.json')!=profile[key]:raise AssertionError('Prepared independent input/reference changed')


def run(root):
    out=root/IDENTITY;file=out/'plan.json';plan=read(file)
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal numerical result')
    activate(root,file,plan);execution=CheckedExecution(root,file,out/'execution');cpu=cf.ThreadPoolExecutor(2)
    result=dict(status='STOPPED_HEAD_FREE_FIXED_TWO_NUMERICAL_QUALIFICATION',records={})
    def interrupted(number,frame):execution.failed.set();raise KeyboardInterrupt('Owned numerical qualification interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def native(trial,mode,parent=None):
        name=phase_name(trial,mode);entry=plan['profiles']['fixed2']
        opts=['--qualification-plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--qualification-sha',execution.sha,
            '--profile','fixed2','--comm-port',str(plan['root_comm_ports'][name]),
            '--preparation-root','/work/instance-nvme/'+entry['preparation_directory'],
            '--corpus','/work/instance-nvme/natural-append-6500m-v2/source.json',
            '--packed-corpus','/work/instance-nvme/natural-append-6500m-v2/packed',
            '--arm','poisson2','--accumulation','16','--precision','fp16','--mode',mode]
        if parent:opts+=['--resume','/work/instance-nvme/runs/'+parent[0]+'/after-120-full-state.pt','--resume-sha',parent[1]['checkpoint_sha256']]
        row=execution.native(name,'scripts.qualify_baseline_managed',opts,'CAPTURED_PARCAE_TRAINING_QUALIFICATION')
        return name,row
    def audit(trial,warm,replay=None):
        mode='replay' if replay else 'audit';target=out/('replay' if replay else 'cpu')/trial
        opts=['--root',str(root),'--plan',str(file),'--plan-sha',execution.sha,'--profile','fixed2','--mode',mode,
            '--warm',str(root/'runs'/warm[0]),'--upstream',str(root/'upstream-parcae-69284c1'),'--output',str(target)]
        if replay:opts+=['--replay',str(root/'runs'/replay[0]),'--gold',str(out/'cpu'/trial)]
        return execution.cpu(trial+'-'+mode,'scripts.audit_baseline_native',opts,target/'report.json',
            'PASS_BASELINE_NAMED_REPLAY_AND_GLOBAL_GRADIENT' if replay else 'PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT')
    def trial_run(trial):
        warm=native(trial,'warm');future=cpu.submit(audit,trial,warm)
        future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
        replay=native(trial,'replay',warm);independent=future.result();restored=audit(trial,warm,replay)
        chips=read(root/'snapshots'/warm[0]/'launch.json')['physical_chips']
        return dict(warm=warm[1],replay=replay[1],independent_cpu=independent,restored=restored,warm_physical_chips=chips)
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_NUMERICAL_ONLY',maximum_chips=16,scientific_fit_admitted=False))
        phase=plan['fixture_phase']
        fixture=execution.native(phase,'scripts.qualify_rejected_state_native_fixture',
            ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
             '--plan-sha',execution.sha,'--comm-port',str(plan['root_comm_ports'][phase])],FIXTURE_STATUS)
        fixture_gate(fixture,execution.sha);result['rejected_state_fixture']=fixture
        write_state(PUBLIC/'rejected-state-fixture.json',fixture)
        with cf.ThreadPoolExecutor(2) as pool:
            jobs={trial:pool.submit(trial_run,trial) for trial in sorted(TRIALS)}
            for f in jobs.values():f.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
            for trial,future in jobs.items():
                result['records'][trial]=future.result();write_state(PUBLIC/'progress.json',result)
        result.update(status=STATUS,plan_sha256=execution.sha,actual_qualification_optimizer_updates=244,
            research_optimizer_updates=0,scientific_fit_admitted=False,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        validate_report(result,plan);execution.check()
    except BaseException as error:
        execution.failed.set();result.update(status='STOPPED_HEAD_FREE_FIXED_TWO_NUMERICAL_QUALIFICATION',error=str(error),traceback=traceback.format_exc())
        write_state(out/'initiating-failure.json',result);raise
    finally:
        execution.failed.set();cpu.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=execution.sha,scope=plan['scope'],scientific_fit_admitted=False,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0,scientific_fit_admitted=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    if not (a.root/IDENTITY/'plan.json').exists():prepare(a.root)
    if a.run:run(a.root)


if __name__=='__main__':main()
