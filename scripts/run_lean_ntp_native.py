"""Wait without chips,then conditionally qualify the prepared lean operators."""
import argparse
import concurrent.futures as cf
import fcntl
import json
from pathlib import Path
import signal
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.fixed_two_curve import stop_on_failed_future
from compute_trainium.future_span_execution import Execution
from compute_trainium.lean_ntp_admission import passes_development_gate
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_fixed_control_replication import reserves

IDENTITY='lean-ntp-native-v1'
QUEUE='lean-ntp-native-queue-v1'
PREP='lean-ntp-native-preparation-v1'
PUBLIC=Path('results/lean_ntp_native_20261026/native')
DOC='docs/LEAN_NTP_NATIVE_ADMISSION_20261026.md'


def read(file):return json.loads(file.read_text())


def queue_plan(root):
    file=root/QUEUE/'plan.json'
    if file.exists():return read(file)
    prep_file=root/PREP/'plan.json';prep=read(prep_file);ready=read(root/PREP/'report.json')
    parent=root/'fixed-control-replication-main-v1/plan.json';clock=read(parent.parent/'execution/execution-start.json')
    if ready['status']!='PASS_LEAN_NTP_NATIVE_PREPARATION_ONLY' or ready['plan_sha256']!=file_sha256(prep_file):
        raise ValueError('Complete unchanged lean CPU preparation required')
    sources=dict(prep['source_sha256'])
    for file_name in ['scripts/run_lean_ntp_native.py','scripts/qualify_lean_ntp_managed.py',
                      'scripts/audit_lean_ntp_replay.py','tests/test_lean_native_managed.py',DOC]:
        sources[file_name]=file_sha256(file_name)
    if any(file_sha256(p)!=sha for p,sha in sources.items()):raise AssertionError('Prepared sources changed')
    plan=dict(identity=QUEUE,registered_unix=time.time(),source_sha256=sources,
              preparation_plan_sha256=file_sha256(prep_file),preparation_report_sha256=file_sha256(root/PREP/'report.json'),
              parent_plan_sha256=file_sha256(parent),wait_deadline_unix=clock['deadline_unix']+600,
              assigned_chips_while_waiting=0,qualification_seconds_cap=3600,
              test_accessed=False,terminal_alpha_spent=0,scope=Path(DOC).read_text())
    file.parent.mkdir();PUBLIC.mkdir(parents=True);write_state(file,plan);write_state(PUBLIC/'queue-plan.json',plan)
    return plan


def frozen(root,queue):
    if (any(file_sha256(p)!=sha for p,sha in queue['source_sha256'].items())
            or file_sha256(root/PREP/'plan.json')!=queue['preparation_plan_sha256']
            or file_sha256(root/PREP/'report.json')!=queue['preparation_report_sha256']
            or file_sha256(root/'fixed-control-replication-main-v1/plan.json')!=queue['parent_plan_sha256']):
        raise AssertionError('Frozen qualification queue source or prerequisite changed')


def register(root,queue,parent_report):
    if managed_running():raise RuntimeError('Entire predecessor must release its native owners')
    if not passes_development_gate(parent_report):raise ValueError('Development gate does not pass')
    reserves(root)
    import shutil
    if shutil.disk_usage(root).free<65*1024**3:raise RuntimeError('Projected qualification storage plus protected reserve required')
    prep=read(root/PREP/'plan.json');parent=read(root/'fixed-control-replication-main-v1/plan.json')
    if (set(parent_report['scores'])!=set(parent['cases']) or set(parent_report['gradient_gates'])!=set(parent['cases'])
            or parent_report['plan_sha256']!=queue['parent_plan_sha256']
            or any(g['independent_cpu']['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
                   for g in parent_report['gradient_gates'].values())):
        raise AssertionError('Every frozen endpoint and numerical gate must complete')
    slower=max(parent['measured_budget']['warm_seconds_per_update'].values())
    estimate=244*slower*1.5/2+4*300/2+2*1200/2+2*120/2+300
    if estimate>3600:raise RuntimeError('Complete numerical qualification does not fit its one-hour cap')
    phases={}
    for label,arm in [('one','fixed1'),('fixed2','poisson2')]:
        for mode in ['warm','replay']:
            name=IDENTITY+'-'+label+'-'+mode
            phases[name]=dict(label=label,arm=arm,mode=mode,parent=IDENTITY+'-'+label+'-warm' if mode=='replay' else None)
    worker,host=split_snapshot_sources(queue['source_sha256'])
    plan=dict(identity=IDENTITY,purpose='lean_ntp_native_qualification_only',registered_unix=time.time(),
        source_sha256=queue['source_sha256'],worker_source_sha256=worker,host_only_source_sha256=host,
        preparation_directory=PREP,preparation_plan_sha256=queue['preparation_plan_sha256'],
        original_preparation_sha256=prep['reference_preparation_plan_sha256'],
        parent_report_sha256=file_sha256(root/'fixed-control-replication-main-v1/report.json'),
        phases=phases,root_comm_ports=reserve_ports(root,list(phases)),seconds_limit=3600,
        phase_seconds_limit=900,cpu_phase_seconds_limit=1200,
        measured_budget=dict(slower_full_head_seconds_per_update=slower,predicted_seconds_with_margin=estimate,cap_seconds=3600),
        test_accessed=False,terminal_alpha_spent=0,scientific_main_admitted=False,scope=queue['scope'])
    out=root/IDENTITY;out.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan


class CheckedExecution(Execution):
    def check(self):
        super().check();reserves(self.root)
        for path,sha in [(self.root/PREP/'plan.json',self.plan['preparation_plan_sha256']),
                         (self.root/'parcae-low-depth-qualification-v1/plan.json',self.plan['original_preparation_sha256']),
                         (self.root/'fixed-control-replication-main-v1/report.json',self.plan['parent_report_sha256'])]:
            if file_sha256(path)!=sha:raise AssertionError('Frozen numerical or development prerequisite changed')


def run(root,*,resume=False):
    out=root/IDENTITY;file=out/'plan.json';plan=read(file)
    if (out/'report.json').exists():
        existing=read(out/'report.json')
        if existing['status']=='PASS_LEAN_NTP_NATIVE_NUMERICAL_AND_REPLAY_QUALIFICATION':return existing
        raise FileExistsError('Preserve terminal qualification;explicit recovery required')
    execution=CheckedExecution(root,file,out/'execution',resume=resume);cpu=cf.ThreadPoolExecutor(2)
    records={};result=dict(status='STOPPED_LEAN_NTP_NATIVE_QUALIFICATION')
    def interrupted(number,frame):execution.failed.set();raise KeyboardInterrupt('Lean qualification interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}

    def native(label,mode,parent=None):
        name=IDENTITY+'-'+label+'-'+mode;spec=plan['phases'][name]
        opts=['--qualification-plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--qualification-sha',execution.sha,
              '--comm-port',str(plan['root_comm_ports'][name]),
              '--preparation-root','/work/instance-nvme/'+PREP,
              '--corpus','/work/instance-nvme/natural-append-6500m-v2/source.json',
              '--packed-corpus','/work/instance-nvme/natural-append-6500m-v2/packed',
              '--arm',spec['arm'],'--precision','fp16','--accumulation','16','--mode',mode]
        if parent:opts+=['--resume','/work/instance-nvme/runs/'+parent[0]+'/after-120-full-state.pt',
                         '--resume-sha',parent[1]['checkpoint_sha256']]
        row=execution.native(name,'scripts.qualify_lean_ntp_managed',opts,'CAPTURED_PARCAE_TRAINING_QUALIFICATION')
        if (row['managed_qualification_plan_sha256']!=execution.sha
                or row['signature']['preparation_sha256']!=plan['preparation_plan_sha256']
                or not row['qualification_only'] or not row['all_replica_model_optimizer_states_equal']):
            raise AssertionError('Exact lean numerical identity/replicas required')
        return name,row

    def audit(label,warm):
        target=out/'cpu'/label
        opts=['--root',str(root),'--primary',str(root/'runs'/warm[0]),
              '--upstream',str(root/'upstream-parcae-69284c1'),'--output',str(target)]
        return execution.cpu(label+'-independent','scripts.audit_parcae_precise_training_point',opts,
                             target/'report.json','PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT')

    def case(label):
        warm=native(label,'warm');future=cpu.submit(audit,label,warm)
        future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
        replay=native(label,'replay',warm);independent=future.result();target=out/'replay'/label
        opts=['--plan',str(file),'--plan-sha',execution.sha,'--warm',str(root/'runs'/warm[0]),
              '--replay',str(root/'runs'/replay[0]),'--gold',str(out/'cpu'/label),'--output',str(target)]
        recovered=execution.cpu(label+'-replay','scripts.audit_lean_ntp_replay',opts,target/'report.json',
                                'PASS_LEAN_NTP_NAMED_REPLAY_AND_GLOBAL_GRADIENT')
        return dict(label=label,warm=warm[1],replay=replay[1],independent_cpu=independent,recovery=recovered)

    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_LEAN_NTP_NUMERICAL_QUALIFICATION',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as workers:
            futures={workers.submit(case,label):label for label in ['one','fixed2']}
            for future in futures:future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
            for future in cf.as_completed(futures):
                row=future.result();records[row['label']]=row;write_state(PUBLIC/'progress.json',dict(records=records))
        if set(records)!={'one','fixed2'}:raise AssertionError('Both numerical profiles must finish')
        updates=sum(r['warm']['actual_optimizer_calls_per_replica']+r['replay']['actual_optimizer_calls_per_replica'] for r in records.values())
        if updates!=244 or any(r['warm']['warm_updates']!=100 for r in records.values()):raise AssertionError('Complete timing/update accounting required')
        execution.check();result=dict(status='PASS_LEAN_NTP_NATIVE_NUMERICAL_AND_REPLAY_QUALIFICATION',
            records=records,actual_qualification_optimizer_updates=updates,research_optimizer_updates=0,
            scientific_main_admitted=False,scope=plan['scope'])
    except BaseException as error:execution.failed.set();result.update(error=str(error),records=records);raise
    finally:
        execution.failed.set();cpu.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=execution.sha,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    queue=queue_plan(a.root)
    if a.prepare_only:return
    directory=a.root/QUEUE
    with (directory/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (directory/'report.json').exists():raise FileExistsError('Preserve the terminal queue receipt')
            if not (a.root/IDENTITY/'execution/execution-start.json').exists() and time.time()>queue['wait_deadline_unix']:
                raise TimeoutError('Original latest admission time elapsed')
            if not (a.root/IDENTITY/'plan.json').exists():
                while True:
                    frozen(a.root,queue)
                    if time.time()>queue['wait_deadline_unix']:raise TimeoutError('Original latest admission time elapsed')
                    file=a.root/'fixed-control-replication-main-v1/report.json'
                    if file.exists():
                        result=read(file)
                        if result['status']!='COMPLETE_FIXED_CONTROL_DEVELOPMENT_REPLICATION':raise RuntimeError('Predecessor stopped;no automatic qualification')
                        if not passes_development_gate(result):
                            receipt=dict(status='SKIPPED_LEAN_NATIVE_DEVELOPMENT_GATE',parent_report_sha256=file_sha256(file),
                                assigned_chips=0,optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0)
                            write_state(directory/'report.json',receipt);write_state(PUBLIC/'report.json',receipt);write_state(PUBLIC/'status.json',receipt);return
                        if not managed_running():register(a.root,queue,result);break
                    write_state(PUBLIC/'status.json',dict(status='WAITING_FOR_COMPLETE_DEVELOPMENT_COMPARISON',assigned_chips=0));time.sleep(10)
            run(a.root,resume=a.resume)
            write_state(directory/'report.json',dict(status=read(a.root/IDENTITY/'report.json')['status'],
                qualification_report_sha256=file_sha256(a.root/IDENTITY/'report.json'),assigned_chips=0))
        except BaseException as error:
            receipt=dict(status='STOPPED_LEAN_NATIVE_QUEUE',error=str(error),
                native_admitted=(a.root/IDENTITY/'execution/execution-start.json').exists(),assigned_chips_while_waiting=0,
                test_accessed=False,terminal_alpha_spent=0)
            if not (directory/'report.json').exists():write_state(directory/'report.json',receipt)
            write_state(PUBLIC/'queue-failure.json',receipt);raise


if __name__=='__main__':main()
