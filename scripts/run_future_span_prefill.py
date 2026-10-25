"""Conditional inference-cost queue;never overlaps an unfinished native campaign."""
import argparse
import concurrent.futures as cf
import copy
import json
import signal
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.future_span_prefill import validate_phase
from compute_trainium.future_span_work_control import admission_evidence
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

PREPARATION='future-span-prefill-preparation-v1'
IDENTITY='future-span-prefill-native-v1'
PUBLIC=Path('results/future_span_prefill_20261025')
START=time.time()


def wait_and_register(root):
    prep_file=root/PREPARATION/'plan.json';prep=json.loads(prep_file.read_text());prep_sha=file_sha256(prep_file)
    report_file=root/PREPARATION/'report.json';cpu=json.loads(report_file.read_text())
    if cpu['status']!='PASS_INDEPENDENT_HEAD_REMOVED_PREFILL_REFERENCES' or cpu['plan_sha256']!=prep_sha:
        raise ValueError('Complete independent actual-weight CPU references required')
    report_sha=file_sha256(report_file)
    main_file=root/prep['source_main_plan'];main=json.loads(main_file.read_text())
    main_report=root/'future-span-main-v2/report.json';secondary_file=root/'ntp-secondary-discovery-v1/report.json'
    work_queue=root/'future-span-work-control-queue-v1/plan.json'
    while True:
        if time.time()>prep['wait_deadline_unix']:raise TimeoutError('Original bounded predecessor window expired')
        if (file_sha256(prep_file)!=prep_sha or file_sha256(report_file)!=report_sha
                or file_sha256(main_file)!=prep['source_main_plan_sha256']
                or file_sha256(work_queue)!=prep['work_queue_plan_sha256']
                or any(file_sha256(f)!=sha for f,sha in prep['source_sha256'].items())):
            raise AssertionError('Frozen references,source or predecessor plans changed')
        work_file=root/'future-span-work-control-v1/report.json'
        queue_file=root/'future-span-work-control-queue-v1/report.json'
        for f in [main_report,secondary_file,work_file,queue_file]:
            if f.exists() and json.loads(f.read_text())['status'].startswith('STOPPED'):
                raise RuntimeError('Predecessor stopped;do not admit inference as an automatic recovery')
        if main_report.exists() and secondary_file.exists():
            parent=json.loads(main_report.read_text());secondary=json.loads(secondary_file.read_text())
            if (parent['plan_sha256']!=prep['source_main_plan_sha256']
                    or secondary['parent_report_sha256']!=file_sha256(main_report)
                    or secondary['parent_plan_sha256']!=prep['source_main_plan_sha256']):
                raise AssertionError('Complete primary/secondary source identity differs')
            evidence=admission_evidence(parent,main,secondary)
            terminal=None
            if evidence['eligible'] and work_file.exists():
                work=json.loads(work_file.read_text());wp=root/'future-span-work-control-v1/plan.json'
                if (work['status']!='COMPLETE_WITHIN_FAMILY_WORK_MATCHED_CONTROLS'
                        or work['plan_sha256']!=file_sha256(wp)
                        or json.loads(wp.read_text())['source_main_report_sha256']!=file_sha256(main_report)):
                    raise ValueError('All registered work controls must complete,regardless of quality ranking')
                terminal=work_file
            elif not evidence['eligible'] and queue_file.exists():
                queue=json.loads(queue_file.read_text())
                if queue['status']!='NOT_ADMITTED_NTP_DISCOVERY_GATES_FAILED':
                    raise ValueError('Exact terminal no-admission receipt required')
                terminal=queue_file
            if terminal is not None and not managed_running():break
        write_state(PUBLIC/'status.json',dict(status='WAITING_FOR_COMPLETE_MAIN_AND_TERMINAL_WORK_CONTROL',assigned_chips=0))
        time.sleep(10)
    out=root/IDENTITY
    if out.exists():raise FileExistsError('Existing inference admission needs explicit resume')
    order=list(prep['cases'])
    phases={IDENTITY+'-'+key+f'-r{round_number}':dict(case_id=key,round=round_number)
            for round_number in [1,2] for key in (order if round_number==1 else order[::-1])}
    remaining=prep['budget']['execution_seconds_cap']-cpu['wall_seconds']
    if remaining<=0:raise ValueError('Independent references exhausted the common30-minute budget')
    plan=copy.deepcopy(prep)
    plan.update(identity=IDENTITY,purpose='read_only_trained_prefill_cost',registered_unix=time.time(),
        preparation=PREPARATION,preparation_plan_sha256=prep_sha,preparation_report_sha256=report_sha,
        source_main_report_sha256=file_sha256(main_report),predecessor_terminal_file=str(terminal.relative_to(root)),
        predecessor_terminal_sha256=file_sha256(terminal),phases=phases,root_comm_ports=reserve_ports(root,list(phases)),
        optimizer_updates_allowed=0,seconds_limit=remaining,phase_seconds_limit=660,cpu_phase_seconds_limit=300,
        prior_cpu_seconds=cpu['wall_seconds'],waiting_seconds_before_admission=time.time()-START)
    out.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'native-plan.json',plan)
    return plan


def run(root,plan,*,resume=False):
    out=root/IDENTITY;file=out/'plan.json'
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal inference measurement')
    execution=Execution(root,file,out/'execution',resume=resume);rows=[]
    result=dict(status='STOPPED_READ_ONLY_PREFILL_MEASUREMENT')
    def interrupted(signum,frame):
        execution.failed.set();raise KeyboardInterrupt('Prefill measurement signal '+str(signum))
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def worker(identity,phase):
        try:
            report=execution.native(identity,'scripts.benchmark_future_span_prefill',
                ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
                 '--plan-sha',execution.sha,'--case',phase['case_id'],'--round',str(phase['round']),
                 '--comm-port',str(plan['root_comm_ports'][identity])],
                'PASS_READ_ONLY_NATIVE_PREFILL_TIMING')
            validate_phase(report,plan,phase['case_id'],phase['round'],execution.sha)
            return dict(identity=identity,**report)
        except BaseException:
            execution.failed.set();raise
    try:
        if (file_sha256(root/plan['predecessor_terminal_file'])!=plan['predecessor_terminal_sha256']
                or file_sha256(root/'future-span-main-v2/report.json')!=plan['source_main_report_sha256']):
            raise AssertionError('Original terminal prerequisites changed')
        for number in [1,2]:
            with cf.ThreadPoolExecutor(2) as pool:
                jobs=[pool.submit(worker,key,value) for key,value in plan['phases'].items() if value['round']==number]
                for future in jobs:
                    rows.append(future.result());write_state(PUBLIC/'native-progress.json',dict(records=rows))
        if {r['identity'] for r in rows}!=set(plan['phases']) or len(rows)!=4:
            raise AssertionError('Both complete paired timing rounds required')
        ratios={}
        for number in [1,2]:
            selected=[r for r in rows if r['round']==number]
            one=next(r for r in selected if plan['cases'][r['case_id']]['source_case']['arm']=='one')
            recurrent=next(r for r in selected if plan['cases'][r['case_id']]['source_case']['arm']=='recurrent')
            values={r['depth']:r for r in recurrent['records']};base=one['records'][0]['timed_forward_median_seconds']
            ratios[number]=dict(recurrent_depth2_over_its_depth1_latency=values[2]['timed_forward_median_seconds']/values[1]['timed_forward_median_seconds'],
                recurrent_depth3_over_its_depth1_latency=values[3]['timed_forward_median_seconds']/values[1]['timed_forward_median_seconds'],
                recurrent_depth2_over_trained_one_pass_latency=values[2]['timed_forward_median_seconds']/base,
                depth2_over_depth1_forward_matrix_ops=values[2]['forward_matrix_ops_per_call']/values[1]['forward_matrix_ops_per_call'])
        assignment={k:{tuple(r['physical_chips']) for r in rows if r['case_id']==k} for k in plan['cases']}
        result=dict(status='COMPLETE_READ_ONLY_SUPPORTED_DEPTH_PREFILL_MEASUREMENT',records=rows,
            latency_and_matrix_ratios=ratios,each_model_measured_on_both_halves=all(len(v)==2 for v in assignment.values()),
            actual_native_model_forwards=sum(sum(s['actual_forwards'] for s in r['records']) for r in rows),
            nominal_forward_matrix_ops=sum(r['nominal_forward_matrix_ops'] for r in rows))
        execution.check()
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),records=rows);raise
    finally:
        execution.failed.set();execution.close(PUBLIC/'native-execution-costs.json')
        for signum,handler in handlers.items():signal.signal(signum,handler)
        result.update(plan_sha256=execution.sha,prior_cpu_seconds=plan['prior_cpu_seconds'],optimizer_updates=0,
            test_accessed=False,terminal_alpha_spent=0,terminal_success=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    def interrupted(signum,frame):raise KeyboardInterrupt('Prefill wait signal '+str(signum))
    for s in [signal.SIGINT,signal.SIGTERM]:signal.signal(s,interrupted)
    if args.resume:
        run(args.root,json.loads((args.root/IDENTITY/'plan.json').read_text()),resume=True)
    else:
        run(args.root,wait_and_register(args.root))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--root',type=Path);args,_=parser.parse_known_args()
        if args.root and not (args.root/IDENTITY/'plan.json').exists():
            result=dict(status='STOPPED_BEFORE_PREFILL_NATIVE_ADMISSION',error=str(error),assigned_chips=0,
                optimizer_updates=0,test_accessed=False,wall_seconds_including_wait=time.time()-START)
            write_state(PUBLIC/'queue-report.json',result)
        raise
