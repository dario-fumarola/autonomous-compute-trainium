"""Conditional finite remediation of stronger one-core baseline numerics."""
import argparse
import concurrent.futures as cf
import copy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.fixed_two_curve import stop_on_failed_future
from compute_trainium.future_span_execution import Execution
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_fixed_control_replication import reserves

IDENTITY='baseline-numerical-v1'
QUEUE='baseline-numerical-queue-v1'
PUBLIC=Path('results/baseline_numerical_20261026')
DOC='docs/BASELINE_NUMERICAL_QUALIFICATION_20261026.md'


def read(path):return json.loads(path.read_text())


def recovery_gate(report,old_gain):
    if report.get('status')!='COMPLETE_ORIGINAL_ONE_CONTROL_RECOVERY':return False
    if set(report.get('scores',{}))!={'one-s17293-data','one-s17293-work'}:return False
    if set(report.get('gradient_gates',{}))!={'stable-prefix','one-s17293-data','one-s17293-work'}:return False
    if any(g['independent_cpu']['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT' for g in report['gradient_gates'].values()):return False
    gain=report['comparisons']['one-s17293-work']['gain_nats']
    return old_gain>0 and gain>0 and (old_gain+gain)/2>=.01


def queue_plan(root):
    out=root/QUEUE
    if (out/'plan.json').exists():return read(out/'plan.json')
    if out.exists():raise FileExistsError('Preserve partial baseline preparation')
    reserves(root)
    one_dir=root/'lean-ntp-native-preparation-v1';one=read(one_dir/'plan.json')
    ordinary_dir=root/'ordinary8-native-preparation-v1';ordinary=read(ordinary_dir/'plan.json')
    parity_file=root/'lean-mixing-precision-preparation-v2/report.json';parity=read(parity_file)
    if parity['status']!='PASS_LEAN_FP32_MIXING_CPU_PARITY_ONLY':raise ValueError('Complete corrected cheap CPU parity required')
    if read(root/'ordinary8-native-preparation-v1/report.json')['status']!='PASS_ORDINARY8_NATIVE_INPUT_PREPARATION_ONLY':raise ValueError('Complete ordinary8 inputs required')
    sources={}
    for group in [one['source_sha256'],ordinary['source_sha256'],parity['source_sha256']]:
        for path,sha in group.items():
            if path in sources and sources[path]!=sha:raise AssertionError('Frozen numerical source versions conflict: '+path)
            sources[path]=sha
    for path in ['scripts/run_baseline_numerical.py','scripts/qualify_baseline_managed.py',
                 'scripts/audit_baseline_native.py','compute_trainium/lean_mixing_reference.py',
                 'tests/test_baseline_qualification.py',DOC]:sources[path]=file_sha256(path)
    if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Frozen CPU/worker sources changed')
    recovery=root/'one-control-recovery-v1/plan.json';old=root/'fixed-two-ntp-main-v1/report.json';previous=read(old)
    old_gain=previous['scores']['one-work-2b']['metrics']['weighted_nll']-previous['scores']['fixed-2b']['metrics']['weighted_nll']
    out.mkdir();mix_dir=out/'mixing-preparation';mix_dir.mkdir();mix=copy.deepcopy(one)
    mix.update(identity=QUEUE+'-mixing-inputs',execution_variant='lean_mixing_fp32_adapter_runtime_scalars',
               source_sha256=sources,scope=Path(DOC).read_text())
    mix['worker_source_sha256'],mix['host_only_source_sha256']=split_snapshot_sources(sources)
    mix['cases']=[c for c in mix['cases'] if c['arm']=='poisson2']
    for initial in mix['initial_states'].values():initial['file']=os.path.relpath(one_dir/initial['file'],mix_dir)
    shutil.copyfile(one_dir/'poisson2-a16-depths.npy',mix_dir/'poisson2-a16-depths.npy')
    write_state(mix_dir/'plan.json',mix);write_state(mix_dir/'cpu-report.json',dict(
        status='COMPLETE_PARCAE_TRAINING_QUALIFICATION_PREPARATION',
        scope='Original qualification-only C0 and full-gradient depth2 table,plus completed corrected/head-free mixing CPU parity. No native qualification or scientific state.'))
    original_reference=root/'parcae-low-depth-qualification-v1';reference=read(original_reference/'plan.json')
    profiles={}
    for label,directory,plan,arm,kind,reference_dir,view,parameters in [
        ('one',one_dir,one,'fixed1','family',original_reference,'.',141963840),
        ('mixing',mix_dir,mix,'poisson2','family',original_reference,'.',141963840),
        ('ordinary8',ordinary_dir,ordinary,'ordinary','ordinary',ordinary_dir,'ordinary8-native-preparation-v1/audit-root',141733120)]:
        initial=plan['initial_states'][kind]
        if file_sha256(directory/initial['file'])!=initial['sha256']:raise AssertionError('Original unfitted numerical initializer changed')
        if read(reference_dir/'plan.json')['model']!=plan['model']:raise AssertionError('Independent reference geometry differs')
        profiles[label]=dict(preparation_directory=str(directory.relative_to(root)),preparation_plan_sha256=file_sha256(directory/'plan.json'),
            arm=arm,parameters=parameters,initial_model_digest=initial['model_digest'],
            reference_preparation_directory=str(reference_dir.relative_to(root)),reference_plan_sha256=file_sha256(reference_dir/'plan.json'),
            reference_root_view=view)
    plan=dict(identity=QUEUE,registered_unix=time.time(),source_sha256=sources,profiles=profiles,
        recovery_plan_sha256=file_sha256(recovery),old_report_sha256=file_sha256(old),old_gain=old_gain,
        mixing_cpu_report_sha256=file_sha256(parity_file),
        wait_deadline_unix=read(root/'one-control-recovery-v1/plan.json')['absolute_deadline_unix']+600,
        qualification_seconds_limit=5400,assigned_chips_while_waiting=0,scientific_fit_admitted=False,
        test_accessed=False,terminal_alpha_spent=0,scope=Path(DOC).read_text())
    write_state(out/'plan.json',plan);PUBLIC.mkdir(parents=True,exist_ok=True);write_state(PUBLIC/'queue-plan.json',plan)
    write_state(PUBLIC/'queue-status.json',dict(status='WAITING_ORIGINAL_ONE_RECOVERY',assigned_chips=0));return plan


def frozen(root,queue):
    if any(file_sha256(f)!=sha for f,sha in queue['source_sha256'].items()):raise AssertionError('Frozen baseline queue source changed')
    for profile in queue['profiles'].values():
        for field,key in [('preparation_directory','preparation_plan_sha256'),('reference_preparation_directory','reference_plan_sha256')]:
            if file_sha256(root/profile[field]/'plan.json')!=profile[key]:raise AssertionError('Frozen baseline/reference preparation changed')
    for path,sha in [(root/'one-control-recovery-v1/plan.json',queue['recovery_plan_sha256']),
                     (root/'fixed-two-ntp-main-v1/report.json',queue['old_report_sha256']),
                     (root/'lean-mixing-precision-preparation-v2/report.json',queue['mixing_cpu_report_sha256'])]:
        if file_sha256(path)!=sha:raise AssertionError('Frozen baseline prerequisite changed')


def register(root,queue,recovery):
    frozen(root,queue)
    if not recovery_gate(recovery,queue['old_gain']):raise ValueError('Baseline-remediation gate does not pass')
    if recovery['plan_sha256']!=queue['recovery_plan_sha256']:raise ValueError('Different completed recovery')
    if managed_running():raise RuntimeError('Predecessor native owners must be closed')
    reserves(root)
    if shutil.disk_usage(root).free<75*1024**3:raise RuntimeError('Projected baseline numerical storage plus reserve required')
    original=read(root/'fixed-control-replication-main-v1/plan.json')
    slower=max(original['measured_budget']['warm_seconds_per_update'].values())
    estimate=2*122*slower*2+3*300+2*1200+2*120+300
    if estimate>5400:raise RuntimeError('Complete conservative numerical budget does not fit')
    phases={}
    for profile in queue['profiles']:
        for mode in ['warm','replay']:
            name=IDENTITY+'-'+profile+'-'+mode
            phases[name]=dict(profile=profile,mode=mode,parent=IDENTITY+'-'+profile+'-warm' if mode=='replay' else None)
    worker,host=split_snapshot_sources(queue['source_sha256'])
    plan=dict(identity=IDENTITY,purpose='baseline_native_qualification_only',source_sha256=queue['source_sha256'],
        worker_source_sha256=worker,host_only_source_sha256=host,profiles=queue['profiles'],phases=phases,
        root_comm_ports=reserve_ports(root,list(phases)),seconds_limit=5400,phase_seconds_limit=900,cpu_phase_seconds_limit=1200,
        measured_budget=dict(slowest_previous_warm_seconds_per_update=slower,slowdown_factor=2,
                             predicted_seconds_with_margin=estimate,cap_seconds=5400),
        recovery_report_sha256=file_sha256(root/'one-control-recovery-v1/report.json'),
        queue_plan_sha256=file_sha256(root/QUEUE/'plan.json'),scientific_fit_admitted=False,
        test_accessed=False,terminal_alpha_spent=0,scope=queue['scope'])
    out=root/IDENTITY;out.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan);return plan


class CheckedExecution(Execution):
    def check(self):
        super().check();reserves(self.root)
        queue=read(self.root/QUEUE/'plan.json');frozen(self.root,queue)
        if file_sha256(self.root/'one-control-recovery-v1/report.json')!=self.plan['recovery_report_sha256']:
            raise AssertionError('Completed One recovery changed')


def run(root):
    out=root/IDENTITY;file=out/'plan.json';plan=read(file)
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal baseline qualification')
    execution=CheckedExecution(root,file,out/'execution');cpu=cf.ThreadPoolExecutor(2);records={}
    result=dict(status='STOPPED_BASELINE_NUMERICAL_QUALIFICATION')
    def interrupted(number,frame):execution.failed.set();raise KeyboardInterrupt('Owned baseline qualification interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def native(profile,mode,parent=None):
        name=IDENTITY+'-'+profile+'-'+mode;entry=plan['profiles'][profile]
        opts=['--qualification-plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--qualification-sha',execution.sha,
            '--profile',profile,'--comm-port',str(plan['root_comm_ports'][name]),
            '--preparation-root','/work/instance-nvme/'+entry['preparation_directory'],
            '--corpus','/work/instance-nvme/natural-append-6500m-v2/source.json',
            '--packed-corpus','/work/instance-nvme/natural-append-6500m-v2/packed',
            '--arm',entry['arm'],'--accumulation','16','--precision','fp16','--mode',mode]
        if parent:opts+=['--resume','/work/instance-nvme/runs/'+parent[0]+'/after-120-full-state.pt','--resume-sha',parent[1]['checkpoint_sha256']]
        row=execution.native(name,'scripts.qualify_baseline_managed',opts,'CAPTURED_PARCAE_TRAINING_QUALIFICATION')
        if (row['managed_qualification_plan_sha256']!=execution.sha or row['qualification_profile']!=profile
                or row['signature']['preparation_sha256']!=entry['preparation_plan_sha256']
                or not row['qualification_only'] or not row['all_replica_model_optimizer_states_equal']):raise AssertionError('Baseline numerical identity/replicas differ')
        return name,row
    def audit(profile,warm,*,replay=None):
        mode='replay' if replay else 'audit';target=out/('replay' if replay else 'cpu')/profile
        opts=['--root',str(root),'--plan',str(file),'--plan-sha',execution.sha,'--profile',profile,'--mode',mode,
              '--warm',str(root/'runs'/warm[0]),'--upstream',str(root/'upstream-parcae-69284c1'),'--output',str(target)]
        if replay:opts+=['--replay',str(root/'runs'/replay[0]),'--gold',str(out/'cpu'/profile)]
        return execution.cpu(profile+'-'+mode,'scripts.audit_baseline_native',opts,target/'report.json',
            'PASS_BASELINE_NAMED_REPLAY_AND_GLOBAL_GRADIENT' if replay else 'PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT')
    def profile_run(profile):
        warm=native(profile,'warm');future=cpu.submit(audit,profile,warm)
        future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
        replay=native(profile,'replay',warm);independent=future.result();restored=audit(profile,warm,replay=replay)
        return dict(warm=warm[1],replay=replay[1],independent_cpu=independent,restored=restored)
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_BASELINE_NUMERICAL_QUALIFICATION',maximum_chips=16))
        with cf.ThreadPoolExecutor(3) as workers:
            futures={workers.submit(profile_run,p):p for p in plan['profiles']}
            for future in futures:future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
            for future in cf.as_completed(futures):
                records[futures[future]]=future.result();write_state(PUBLIC/'progress.json',dict(records=records))
        updates=sum(x['warm']['actual_optimizer_calls_per_replica']+x['replay']['actual_optimizer_calls_per_replica'] for x in records.values())
        if updates!=366 or set(records)!=set(plan['profiles']) or any(x['warm']['warm_updates']!=100 for x in records.values()):
            raise AssertionError('Complete qualification update/timing inventory differs')
        execution.check();result=dict(status='PASS_THREE_BASELINE_NUMERICAL_AND_REPLAY_PROFILES',records=records,
            actual_qualification_optimizer_updates=updates,research_optimizer_updates=0,scientific_fit_admitted=False,scope=plan['scope'])
    except BaseException as error:execution.failed.set();result.update(error=str(error),records=records);raise
    finally:
        execution.failed.set();cpu.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=execution.sha,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    queue=queue_plan(a.root)
    if a.prepare_only:return
    out=a.root/QUEUE
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal baseline queue')
    try:
        while time.time()<queue['wait_deadline_unix']:
            frozen(a.root,queue);file=a.root/'one-control-recovery-v1/report.json'
            if file.exists():
                recovery=read(file)
                if not recovery_gate(recovery,queue['old_gain']):raise RuntimeError('Original One recovery failed or baseline-remediation effect gate not met')
                if managed_running():time.sleep(5);continue
                register(a.root,queue,recovery);write_state(out/'report.json',dict(status='ADMITTED_BASELINE_NUMERICAL_ONLY',scientific_fit_admitted=False))
                run(a.root);return
            time.sleep(10)
        raise TimeoutError('Original baseline qualification latest-admission window expired')
    except BaseException as error:
        write_state(out/'failure.json',dict(status='STOPPED_BASELINE_NUMERICAL_QUEUE',error=str(error),
            native_admitted=(a.root/IDENTITY/'plan.json').exists(),scientific_fit_admitted=False,test_accessed=False,terminal_alpha_spent=0));raise


if __name__=='__main__':main()
