"""Reuse valid numerical work; finish only the missing replay/ordinary checks."""
import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time
import traceback

from compute_trainium.execution_isolation import managed_running
from compute_trainium.fixed_two_curve import stop_on_failed_future
from compute_trainium.future_span_execution import Execution
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_baseline_numerical import recovery_gate,read
from scripts.run_fixed_control_replication import reserves

IDENTITY='baseline-numerical-recovery-v1'
PUBLIC=Path('results/baseline_numerical_recovery_20261026')
DOC='docs/BASELINE_NUMERICAL_RECOVERY_20261026.md'


def register(root):
    out=root/IDENTITY
    if (out/'plan.json').exists():return read(out/'plan.json')
    reserves(root)
    if managed_running():raise RuntimeError('All stopped predecessor native owners must be closed')
    if shutil.disk_usage(root).free<55*1024**3 or shutil.disk_usage('/').free<37*1024**3:raise RuntimeError('Complete projected recovery space plus protected reserves required')
    parent_file=root/'baseline-numerical-v1/plan.json';old=read(parent_file);old_sha=file_sha256(parent_file)
    if read(parent_file.parent/'report.json')['status']!='STOPPED_BASELINE_NUMERICAL_QUALIFICATION':raise ValueError('Preserved stopped predecessor required')
    one_recovery=read(root/'one-control-recovery-v1/report.json')
    queue=read(root/'baseline-numerical-queue-v1/plan.json')
    if not recovery_gate(one_recovery,queue['old_gain']):raise ValueError('Original development admission gate must still pass')
    sources=dict(old['source_sha256'])
    for name in ['scripts/run_baseline_numerical_recovery.py','scripts/audit_baseline_recovery_replay.py',
                 'compute_trainium/native_inode_projection.py','tests/test_native_inode_projection.py',
                 'scripts/reclaim_closed_snapshot_bytecode.py','tests/test_closed_bytecode_reclamation.py',DOC]:sources[name]=file_sha256(name)
    if any(file_sha256(name)!=sha for name,sha in sources.items()):raise AssertionError('Exact original/new source lists changed')
    inherited={};timings=[]
    for profile in ['one','mixing']:
        name='baseline-numerical-v1-'+profile+'-warm';folder=root/'runs'/name;native=read(folder/'report.json')
        gold_dir=root/f'baseline-cpu-recovery-{profile}-v1';gold=read(gold_dir/'report.json')
        if (native['managed_qualification_plan_sha256']!=old_sha or native['qualification_profile']!=profile
                or native['signature']['preparation_sha256']!=old['profiles'][profile]['preparation_plan_sha256']
                or native['arm']!=old['profiles'][profile]['arm']
                or native['initial_model_digest']!=old['profiles'][profile]['initial_model_digest']
                or native['warm_updates']!=100 or native['actual_optimizer_calls_per_replica']!=121
                or not native['all_replica_model_optimizer_states_equal'] or not native['qualification_only']
                or gold['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT'
                or not gold['first_actual_update']['passes_registered_gate']
                or gold['checkpoint_sha256']!=native['checkpoint_sha256']):raise ValueError('Complete immutable warm and independent reference required')
        for file,sha in [(folder/'after-120-full-state.pt',native['checkpoint_sha256']),
                         (folder/'after-121-model-optimizer.pt',native['final_checkpoint_sha256']),
                         (gold_dir/'cpu-global.pt',gold['cpu_global_gradient_sha256'])]:
            if file_sha256(file)!=sha:raise AssertionError('Qualified state/reference bytes changed')
        if read(root/'snapshots'/name/'closure.json')['exit_code']!=0:raise AssertionError('Successful warm closure required')
        inherited[profile]=dict(warm_identity=name,warm_report_sha256=file_sha256(folder/'report.json'),
            checkpoint_sha256=native['checkpoint_sha256'],managed_plan_sha256=old_sha,
            gold_directory=str(gold_dir.relative_to(root)),gold_report_sha256=file_sha256(gold_dir/'report.json'),
            gold_gradient_sha256=gold['cpu_global_gradient_sha256'])
        timings.append(native['warm_seconds']/100)
    phases={}
    for profile in ['one','mixing','ordinary8']:
        modes=['warm','replay'] if profile=='ordinary8' else ['replay']
        for mode in modes:
            name=IDENTITY+'-'+profile+'-'+mode
            phases[name]=dict(profile=profile,mode=mode,parent=(IDENTITY+'-ordinary8-warm' if profile=='ordinary8'
                else inherited[profile]['warm_identity']) if mode=='replay' else None)
    inode=project_snapshot_inodes(Path.cwd(),len(phases))
    if os.statvfs(root).f_favail<inode['required_free_inodes']:raise RuntimeError('Projected source copies/native artifacts would breach inode reserve')
    slower=max(timings);estimate=124*slower*2+3*300+1200+2*120+300
    if estimate>3600:raise RuntimeError('Complete measured remaining numerical work does not fit one hour')
    worker,host=split_snapshot_sources(sources)
    plan=dict(identity=IDENTITY,purpose='baseline_native_qualification_only',registered_unix=time.time(),
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        profiles=old['profiles'],phases=phases,inherited=inherited,root_comm_ports=reserve_ports(root,list(phases)),
        seconds_limit=3600,phase_seconds_limit=900,cpu_phase_seconds_limit=1200,
        projected_inodes=inode,measured_budget=dict(slowest_completed_warm_seconds_per_update=slower,
            predicted_seconds_with_margin=estimate,cap_seconds=3600),
        original_plan=str(parent_file.relative_to(root)),original_plan_sha256=old_sha,
        original_report_sha256=file_sha256(parent_file.parent/'report.json'),
        one_recovery_report_sha256=file_sha256(root/'one-control-recovery-v1/report.json'),
        new_qualification_updates=124,inherited_qualification_updates=242,scientific_fit_admitted=False,
        test_accessed=False,terminal_alpha_spent=0,scope=Path(DOC).read_text())
    out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=True);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    write_state(PUBLIC/'status.json',dict(status='REGISTERED_RECOVERY_NOT_STARTED',assigned_chips=0));return plan


class CheckedExecution(Execution):
    def check(self):
        try:
            super().check();reserves(self.root)
            if file_sha256(self.root/self.plan['original_plan'])!=self.plan['original_plan_sha256']:raise AssertionError('Original numerical plan changed')
            if file_sha256(self.root/'one-control-recovery-v1/report.json')!=self.plan['one_recovery_report_sha256']:raise AssertionError('Audited development recovery changed')
        except BaseException as error:
            if hasattr(self,'out'):
                write_state(self.out/('guard-failure-'+str(threading.get_ident())+'.json'),dict(
                    error=str(error),kind=type(error).__name__,unix=time.time(),
                    free_inodes=os.statvfs(self.root).f_favail,already_stopping=self.failed.is_set(),traceback=traceback.format_exc()))
            raise


def run(root):
    out=root/IDENTITY;file=out/'plan.json';plan=read(file)
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal numerical recovery')
    reserves(root)
    if managed_running():raise RuntimeError('No unrelated owner at numerical admission')
    if os.statvfs(root).f_favail<plan['projected_inodes']['required_free_inodes']:raise RuntimeError('Projected inode budget no longer fits')
    execution=CheckedExecution(root,file,out/'execution');cpu=cf.ThreadPoolExecutor(2);records={};failures=[];lock=threading.Lock()
    result=dict(status='STOPPED_BASELINE_NUMERICAL_RECOVERY')
    def interrupted(number,frame):
        write_state(out/'signal.json',dict(signal=number,unix=time.time()));execution.failed.set();raise KeyboardInterrupt('Owned numerical recovery interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def observed(label,function,*args,**kwargs):
        try:return function(*args,**kwargs)
        except BaseException as error:
            with lock:
                failures.append(dict(phase=label,unix=time.time(),error=str(error),kind=type(error).__name__,traceback=traceback.format_exc()))
                write_state(out/'phase-failures.json',dict(failures=failures))
            execution.failed.set();raise
    def native(profile,mode,warm=None):
        name=IDENTITY+'-'+profile+'-'+mode;entry=plan['profiles'][profile]
        opts=['--qualification-plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--qualification-sha',execution.sha,
            '--profile',profile,'--comm-port',str(plan['root_comm_ports'][name]),
            '--preparation-root','/work/instance-nvme/'+entry['preparation_directory'],
            '--corpus','/work/instance-nvme/natural-append-6500m-v2/source.json',
            '--packed-corpus','/work/instance-nvme/natural-append-6500m-v2/packed',
            '--arm',entry['arm'],'--accumulation','16','--precision','fp16','--mode',mode]
        if warm:opts+=['--resume','/work/instance-nvme/runs/'+warm[0]+'/after-120-full-state.pt','--resume-sha',warm[1]['checkpoint_sha256']]
        row=execution.native(name,'scripts.qualify_baseline_managed',opts,'CAPTURED_PARCAE_TRAINING_QUALIFICATION')
        if (row['managed_qualification_plan_sha256']!=execution.sha or row['qualification_profile']!=profile
                or row['signature']['preparation_sha256']!=entry['preparation_plan_sha256']
                or not row['all_replica_model_optimizer_states_equal'] or not row['qualification_only']):raise AssertionError('Exact numerical phase/replica identity differs')
        return name,row
    def replay_audit(profile,warm,replay,gold):
        target=out/'replay'/profile
        opts=['--root',str(root),'--plan',str(file),'--plan-sha',execution.sha,'--profile',profile,
              '--warm',str(root/'runs'/warm[0]),'--replay',str(root/'runs'/replay[0]),'--gold',str(gold),'--output',str(target)]
        return execution.cpu(profile+'-replay-check','scripts.audit_baseline_recovery_replay',opts,target/'report.json',
                             'PASS_BASELINE_RECOVERY_NAMED_REPLAY_AND_GLOBAL_GRADIENT')
    def old_profile(profile):
        entry=plan['inherited'][profile];folder=root/'runs'/entry['warm_identity']
        if file_sha256(folder/'report.json')!=entry['warm_report_sha256']:raise AssertionError('Inherited successful warm report changed')
        gold=root/entry['gold_directory']
        if file_sha256(gold/'report.json')!=entry['gold_report_sha256'] or file_sha256(gold/'cpu-global.pt')!=entry['gold_gradient_sha256']:
            raise AssertionError('Pinned inherited full-global reference changed')
        warm=(entry['warm_identity'],read(folder/'report.json'));replay=observed(profile+'-native-replay',native,profile,'replay',warm)
        checked=observed(profile+'-replay-check',replay_audit,profile,warm,replay,gold)
        return dict(warm=warm[1],replay=replay[1],independent_cpu=read(gold/'report.json'),restored=checked,inherited_warm=True)
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_INHERITED_BASELINE_REPLAYS',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as pool:
            futures={pool.submit(old_profile,p):p for p in ['one','mixing']}
            for future in futures:future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
            for future in cf.as_completed(futures):records[futures[future]]=future.result();write_state(PUBLIC/'progress.json',dict(records=records))
        write_state(PUBLIC/'status.json',dict(status='RUNNING_ORDINARY8_NUMERICAL_QUALIFICATION',maximum_chips=8))
        warm=observed('ordinary8-warm',native,'ordinary8','warm')
        target=out/'cpu/ordinary8';opts=['--root',str(root),'--plan',str(file),'--plan-sha',execution.sha,
            '--profile','ordinary8','--mode','audit','--warm',str(root/'runs'/warm[0]),
            '--upstream',str(root/'upstream-parcae-69284c1'),'--output',str(target)]
        future=cpu.submit(observed,'ordinary8-reference',execution.cpu,'ordinary8-reference','scripts.audit_baseline_native',opts,
                          target/'report.json','PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT')
        future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
        replay=observed('ordinary8-native-replay',native,'ordinary8','replay',warm);gold=future.result()
        checked=observed('ordinary8-replay-check',replay_audit,'ordinary8',warm,replay,target)
        records['ordinary8']=dict(warm=warm[1],replay=replay[1],independent_cpu=gold,restored=checked,inherited_warm=False)
        actual=sum(r['replay']['actual_optimizer_calls_per_replica']+(0 if r['inherited_warm'] else r['warm']['actual_optimizer_calls_per_replica']) for r in records.values())
        if actual!=124 or any(r['warm']['warm_updates']!=100 for r in records.values()):raise AssertionError('Complete reused/new qualification work differs')
        execution.check();result=dict(status='PASS_THREE_BASELINE_NUMERICAL_RECOVERY_PROFILES',records=records,
            new_qualification_optimizer_updates=124,inherited_qualification_optimizer_updates=242,
            total_successful_qualification_optimizer_updates=366,research_optimizer_updates=0,scientific_fit_admitted=False,scope=plan['scope'])
    except BaseException as error:execution.failed.set();result.update(error=str(error),records=records,phase_failures=failures);raise
    finally:
        execution.failed.set();cpu.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=execution.sha,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    register(a.root)
    if not a.prepare_only:run(a.root)


if __name__=='__main__':main()
