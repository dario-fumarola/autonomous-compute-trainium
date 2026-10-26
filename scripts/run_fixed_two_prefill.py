"""Wait without chips, then measure the two frozen primary prediction paths."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.future_span_prefill import validate_phase
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.paired_prefill_round import paired_round
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

QUEUE='fixed-two-prefill-queue-v1'
PREPARATION='fixed-two-prefill-preparation-v1'
IDENTITY='fixed-two-prefill-native-v1'
PUBLIC=Path('results/fixed_two_prefill_20261025')
EXTRAS=['scripts/run_fixed_two_prefill.py','scripts/prepare_fixed_two_prefill_cpu.py',
        'scripts/benchmark_fixed_two_prefill.py','compute_trainium/paired_prefill_round.py',
        'compute_trainium/future_span_prefill.py','compute_trainium/inode_reserve.py',
        'tests/test_paired_prefill_round.py','docs/FIXED_TWO_PREFILL_COST_20261025.md',
        'results/future_span_prefill_20261025/preparation-plan.json',
        'results/future_span_prefill_20261025/preparation-report.json',
        'results/future_span_prefill_20261025/native-progress.json']


class PrefillExecution(Execution):
    def check(self):
        super().check();require_inode_reserve([self.root,Path('/')])


def queue_plan(root):
    directory=root/QUEUE;file=directory/'plan.json'
    if file.exists():return json.loads(file.read_text())
    main_file=root/'fixed-two-ntp-main-v1/plan.json';main=json.loads(main_file.read_text())
    clock=json.loads((root/'fixed-two-ntp-main-v1/execution/execution-start.json').read_text())
    budget=json.loads(Path('results/future_span_prefill_20261025/preparation-plan.json').read_text())['budget']
    if not budget['fits_cap'] or budget['two_rounds_plus_cpu_seconds_allowance']>1800:
        raise ValueError('Measured conservative same-shape budget must fit before admission')
    sources={**main['source_sha256'],**{f:file_sha256(f) for f in EXTRAS}}
    if any(file_sha256(f)!=h for f,h in sources.items()):raise AssertionError('Frozen prerequisite source changed')
    plan=dict(identity=QUEUE,registered_unix=time.time(),source_sha256=sources,
        source_main_plan=str(main_file.relative_to(root)),source_main_plan_sha256=file_sha256(main_file),
        source_main_report='fixed-two-ntp-main-v1/report.json',wait_deadline_unix=clock['deadline_unix']+300,
        cases=['one-work-2b','fixed-2b'],depths={'one-work-2b':[1],'fixed-2b':[2]},
        probe_start=5120*512,windows=64,context=1024,warm_forwards=100,timed_forwards=100,
        post_main_seconds_cap=1800,cpu_seconds_cap=300,native_phase_seconds_cap=660,
        measured_budget=budget,budget_scope='Reuse the conservative measured-training bound for One plus all three recurrent exits;the new fixed candidate times only its trained depth2. No speedup assumption.',
        assigned_chips_while_waiting=0,optimizer_updates_allowed=0,
        scope=Path('docs/FIXED_TWO_PREFILL_COST_20261025.md').read_text(),test_accessed=False,terminal_alpha_spent=0)
    directory.mkdir();PUBLIC.mkdir();write_state(file,plan);write_state(PUBLIC/'queue-plan.json',plan)
    return plan


def wait(root,queue):
    while True:
        if time.time()>queue['wait_deadline_unix']:raise TimeoutError('Original predecessor window expired')
        if (file_sha256(root/queue['source_main_plan'])!=queue['source_main_plan_sha256']
                or any(file_sha256(f)!=h for f,h in queue['source_sha256'].items())):
            raise AssertionError('Frozen queue/source changed')
        report_file=root/queue['source_main_report']
        if report_file.exists():
            report=json.loads(report_file.read_text())
            if report['status']!='COMPLETE_FIXED_TWO_NTP_CURVE':
                raise ValueError('Preserve stopped predecessor; no inference recovery admission')
            if report['plan_sha256']!=queue['source_main_plan_sha256']:
                raise AssertionError('Completed scientific predecessor changed')
            if not managed_running():return report
        write_state(PUBLIC/'status.json',dict(status='WAITING_FOR_COMPLETE_FIXED_CURVE',assigned_chips=0))
        time.sleep(5)


def prepare(root,queue,parent):
    start=time.time();deadline=start+queue['post_main_seconds_cap']
    main_file=root/queue['source_main_plan'];main=json.loads(main_file.read_text());cases={}
    for key in queue['cases']:
        node=key+'-finish'
        capture=next(r for r in parent['training_records'] if r['node']==node)
        trained=capture['report'];case=main['cases'][key]
        if trained['update']!=case['total_updates']:raise AssertionError('Incomplete primary prediction endpoint')
        record=next(r for r in trained['checkpoint_records'] if r['update']==case['total_updates'])
        checkpoint=root/'runs'/capture['identity']/f"checkpoint-u{case['total_updates']}.pt"
        if file_sha256(checkpoint)!=record['checkpoint_sha256']:raise AssertionError('Completed endpoint changed')
        cases[key]=dict(source_case=case,checkpoint=str(checkpoint.relative_to(root)),
            checkpoint_sha256=record['checkpoint_sha256'],full_state_digest=record['full_state_digest'],depths=queue['depths'][key])
    prep=copy.deepcopy(queue);prep.update(identity=PREPARATION,purpose='fixed_two_prefill_cpu_reference',
        model=main['model'],cases=cases,registered_unix=start,absolute_deadline_unix=deadline,
        source_main_report_sha256=file_sha256(root/queue['source_main_report']))
    directory=root/PREPARATION;directory.mkdir();file=directory/'plan.json';write_state(file,prep)
    write_state(PUBLIC/'preparation-plan.json',prep)
    with (directory/'cpu.log').open('xb') as log:
        subprocess.run([os.sys.executable,'-u','-m','scripts.prepare_fixed_two_prefill_cpu',
            '--root',str(root),'--plan',str(file)],check=True,timeout=queue['cpu_seconds_cap'],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'USE_TORCH_XLA':'0','OMP_NUM_THREADS':'12','MKL_NUM_THREADS':'12'})
    report_file=directory/'report.json';cpu=json.loads(report_file.read_text())
    if cpu['status']!='PASS_INDEPENDENT_HEAD_REMOVED_PREFILL_REFERENCES' or cpu['plan_sha256']!=file_sha256(file):
        raise ValueError('Complete independent actual-weight references required')
    write_state(PUBLIC/'preparation-report.json',cpu)
    phases={IDENTITY+'-'+key+f'-r{round_number}':dict(case_id=key,round=round_number)
            for round_number in [1,2] for key in (queue['cases'] if round_number==1 else queue['cases'][::-1])}
    plan=copy.deepcopy(prep);worker,host=split_snapshot_sources(plan['source_sha256'])
    plan.update(identity=IDENTITY,purpose='read_only_trained_prefill_cost',preparation=PREPARATION,
        preparation_plan_sha256=file_sha256(file),preparation_report_sha256=file_sha256(report_file),
        worker_source_sha256=worker,host_only_source_sha256=host,phases=phases,root_comm_ports=reserve_ports(root,list(phases)),
        seconds_limit=int(deadline-time.time()),phase_seconds_limit=queue['native_phase_seconds_cap'],cpu_phase_seconds_limit=300,
        prior_cpu_seconds=cpu['wall_seconds'],waiting_seconds_before_admission=start-queue['registered_unix'])
    if plan['seconds_limit']<2*queue['native_phase_seconds_cap']:
        raise TimeoutError('CPU reference exhausted the bounded native allowance')
    out=root/IDENTITY;out.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'native-plan.json',plan)
    return plan


def run(root,plan,resume=False):
    out=root/IDENTITY
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal inference result')
    execution=PrefillExecution(root,out/'plan.json',out/'execution',resume=resume)
    rows=[];result=dict(status='STOPPED_FIXED_CURVE_PREFILL_MEASUREMENT')
    def interrupted(signum,frame):
        execution.failed.set();raise KeyboardInterrupt('Read-only timing signal '+str(signum))
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def worker(identity,phase):
        row=execution.native(identity,'scripts.benchmark_fixed_two_prefill',
            ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
             '--plan-sha',execution.sha,'--case',phase['case_id'],'--round',str(phase['round']),
             '--comm-port',str(plan['root_comm_ports'][identity])],'PASS_READ_ONLY_NATIVE_PREFILL_TIMING')
        validate_phase(row,plan,phase['case_id'],phase['round'],execution.sha)
        expected_half=0 if phase['case_id']==('one-work-2b' if phase['round']==1 else 'fixed-2b') else 1
        if row['physical_chips']!=list(range(expected_half*8,(expected_half+1)*8)):
            raise AssertionError('Actual assigned half differs from the frozen crossover')
        return dict(identity=identity,**row)
    try:
        require_inode_reserve([root,Path('/')])
        if file_sha256(root/plan['source_main_report'])!=plan['source_main_report_sha256']:
            raise AssertionError('Completed predecessor changed')
        write_state(PUBLIC/'status.json',dict(status='RUNNING_READ_ONLY_FIXED_CURVE_PREFILL',maximum_chips=16))
        for number in [1,2]:
            phases=[(k,v) for k,v in plan['phases'].items() if v['round']==number]
            rows.extend(paired_round(execution,phases,worker));write_state(PUBLIC/'native-progress.json',dict(records=rows))
        ratios=[]
        for number in [1,2]:
            pair={r['case_id']:r['records'][0] for r in rows if r['round']==number}
            a,b=pair['one-work-2b'],pair['fixed-2b']
            ratios.append(dict(round=number,fixed_over_one_median_latency=b['timed_forward_median_seconds']/a['timed_forward_median_seconds'],
                fixed_over_one_forward_matrix_work=b['forward_matrix_ops_per_call']/a['forward_matrix_ops_per_call']))
        result=dict(status='COMPLETE_FIXED_CURVE_PREFILL_MEASUREMENT',records=rows,ratios=ratios,
            each_model_measured_on_both_halves=True,actual_native_model_forwards=sum(r['records'][0]['actual_forwards'] for r in rows),
            nominal_native_forward_matrix_ops=sum(r['nominal_forward_matrix_ops'] for r in rows))
        execution.check()
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),records=rows);raise
    finally:
        execution.failed.set();execution.close(PUBLIC/'native-execution-costs.json')
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=execution.sha,optimizer_updates=0,prior_cpu_seconds=plan['prior_cpu_seconds'],
            post_main_whole_instance_opportunity_chip_hours=(time.time()-plan['registered_unix'])*16/3600,
            test_accessed=False,terminal_alpha_spent=0,terminal_success=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--register-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    def interrupted(signum,frame):raise KeyboardInterrupt('Read-only prefill queue signal '+str(signum))
    for number in [signal.SIGINT,signal.SIGTERM]:signal.signal(number,interrupted)
    if a.resume:
        run(a.root,json.loads((a.root/IDENTITY/'plan.json').read_text()),resume=True);return
    queue=queue_plan(a.root)
    if a.register_only:return
    try:
        run(a.root,prepare(a.root,queue,wait(a.root,queue)))
    except BaseException as error:
        receipt=dict(status='STOPPED_FIXED_PREFILL_QUEUE_OR_ADMISSION',error=str(error),assigned_chips=0,
            native_plan_created=(a.root/IDENTITY/'plan.json').exists(),optimizer_updates=0,test_accessed=False)
        prep=a.root/PREPARATION/'plan.json'
        if prep.exists():
            registered=json.loads(prep.read_text())['registered_unix']
            receipt['post_main_wall_seconds']=time.time()-registered
            receipt['post_main_whole_instance_opportunity_chip_hours']=(time.time()-registered)*16/3600
        write_state(a.root/QUEUE/'report.json',receipt);write_state(PUBLIC/'queue-report.json',receipt)
        if not (PUBLIC/'report.json').exists():write_state(PUBLIC/'status.json',receipt)
        raise
    receipt=dict(status='COMPLETE_FIXED_PREFILL_QUEUE',native_report_sha256=file_sha256(a.root/IDENTITY/'report.json'),
        optimizer_updates=0,assigned_chips=0,test_accessed=False)
    write_state(a.root/QUEUE/'report.json',receipt);write_state(PUBLIC/'queue-report.json',receipt)


if __name__=='__main__':main()
