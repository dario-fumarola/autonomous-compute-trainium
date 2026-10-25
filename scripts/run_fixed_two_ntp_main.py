"""Ready-node queue for six cooled source/work endpoints; no early quality stop."""
import argparse
import concurrent.futures as cf
import fcntl
import json
from pathlib import Path
import signal
import time
import numpy as np

from compute_trainium.fixed_two_curve import validate_curve,stop_on_failed_future
from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.metrics import DocumentScore,paired_group_interval
from compute_trainium.reference_loss_strata import reference_deciles
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.register_fixed_two_ntp_main import IDENTITY,PUBLIC,register


class CurveExecution(Execution):
    def check(self):
        super().check();require_inode_reserve([self.root,Path('/')])


def checkpoint(capture):
    return next(row for row in capture[1]['checkpoint_records'] if row['update']==capture[1]['update'])


def queue_plan(root):
    directory=root/'fixed-two-ntp-main-queue-v1';public=Path('results/fixed_two_small_core_main_queue_20261025');file=directory/'plan.json'
    if file.exists():return directory,public,json.loads(file.read_text())
    tune_file=root/'fixed-two-ntp-tune-v1/plan.json';tune=json.loads(tune_file.read_text());prep=root/'fixed-two-ntp-main-preparation-v1'
    preparation=json.loads((prep/'registration.json').read_text());start=json.loads((root/'fixed-two-ntp-tune-v1/execution/execution-start.json').read_text())
    extras=['scripts/register_fixed_two_ntp_main.py','scripts/run_fixed_two_ntp_main.py','compute_trainium/fixed_two_curve.py','tests/test_fixed_two_curve.py','scripts/evaluate_fixed_two_curve_cpu.py','compute_trainium/reference_loss_strata.py','tests/test_reference_loss_strata.py','docs/FIXED_TWO_NTP_STRATA_20261025.md','docs/FIXED_TWO_SMALL_CORE_MAIN_ADMISSION_20261025.md']
    sources={**tune['source_sha256'],**preparation['source_sha256'],**{f:file_sha256(f) for f in extras}}
    plan=dict(identity=directory.name,registered_unix=time.time(),source_sha256=sources,tune_plan_sha256=file_sha256(tune_file),preparation_manifest_sha256=file_sha256(prep/'manifest.json'),reference_parity_report_sha256=file_sha256(root/'fixed-two-strata-reference-v1/report.json'),wait_deadline_unix=start['deadline_unix']+300,assigned_chips_while_waiting=0,main_seconds_cap=18000,scope='Wait without chips for all fixed TUNE numerical/timing/endpoint gates,then separately freeze/admit the predeclared finite six-endpoint MAIN if its complete measured budget fits. No quality-based LR-grid,exposure or architecture expansion. Original waiting deadline never resets;MAIN receives its own declared five-hour execution clock. Parent and MAIN receipts stay separate;no double-counted reservation interval.',test_accessed=False,terminal_alpha_spent=0)
    directory.mkdir();public.mkdir();write_state(file,plan);write_state(public/'plan.json',plan)
    return directory,public,plan


def wait_for_tune(root):
    directory,public,queue=queue_plan(root)
    while True:
        if time.time()>queue['wait_deadline_unix']:raise TimeoutError('Original MAIN queue admission window elapsed')
        if any(file_sha256(f)!=sha for f,sha in queue['source_sha256'].items()) or file_sha256(root/'fixed-two-ntp-tune-v1/plan.json')!=queue['tune_plan_sha256'] or file_sha256(root/'fixed-two-ntp-main-preparation-v1/manifest.json')!=queue['preparation_manifest_sha256'] or file_sha256(root/'fixed-two-strata-reference-v1/report.json')!=queue['reference_parity_report_sha256']:raise AssertionError('Frozen MAIN queue input/source changed')
        file=root/'fixed-two-ntp-tune-v1/report.json'
        if file.exists():
            report=json.loads(file.read_text())
            if report['status']!='COMPLETE_FIXED_TWO_NTP_TUNING':raise RuntimeError('Parent TUNE stopped or failed;MAIN not admitted')
            if not managed_running():return public
        write_state(public/'status.json',dict(status='WAITING_FOR_COMPLETE_FIXED_TUNE',assigned_chips=0));time.sleep(10)


def run(root,*,resume=False):
    out=root/IDENTITY;plan_file=out/'plan.json';plan=json.loads(plan_file.read_text())
    if (out/'report.json').exists():
        value=json.loads((out/'report.json').read_text())
        if value['status']=='COMPLETE_FIXED_TWO_NTP_CURVE':return value
        raise FileExistsError('Preserve a failed/stopped execution; explicit recovery registration required')
    if file_sha256(root/'fixed-two-ntp-tune-v1/report.json')!=plan['tune_report_sha256'] or file_sha256(root/'fixed-two-ntp-main-preparation-v1/manifest.json')!=plan['preparation_manifest_sha256']:raise AssertionError('Original qualified TUNE/preparation changed')
    validate_curve(plan['cases'],plan['nodes'])
    execution=CurveExecution(root,plan_file,out/'execution',resume=resume);cpu=cf.ThreadPoolExecutor(3);diagnostics=cf.ThreadPoolExecutor(2)
    captures={};readouts={};gates={};work_records=[];diagnostic_handles={}
    result=dict(status='STOPPED_FIXED_TWO_NTP_CURVE')
    def interrupted(number,frame):execution.failed.set();raise KeyboardInterrupt('Fixed curve interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}

    def fit(node_id,parent):
        node=plan['nodes'][node_id];key=node['case'];case=plan['cases'][key];name=IDENTITY+'-'+node_id
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--case',key,'--segment-end',str(node['end']),'--comm-port',str(plan['root_comm_ports'][name])]
        source=None
        if parent:
            source=checkpoint(parent)
            opts+=['--resume','/work/instance-nvme/runs/'+parent[0]+f'/checkpoint-u{source["update"]}.pt','--resume-sha',source['checkpoint_sha256']]
        row=execution.native(name,'scripts.train_future_span_primitives',opts,'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT')
        if row['plan_sha256']!=execution.sha or row['case_id']!=key or row['update']!=node['end'] or row['qualification_only'] or row['objective_weight']!=0.:raise AssertionError('Fixed curve segment identity differs')
        for state in row['checkpoint_records']:
            if not state['all_complete_replica_states_equal'] or file_sha256(root/'runs'/name/f'checkpoint-u{state["update"]}.pt')!=state['checkpoint_sha256']:raise AssertionError('Complete saved replicas/checkpoint differ')
        if parent and (not row['complete_state_on_load_exact'] or not row['cpu_rng_on_load_exact'] or row['recovered_state_digest']!=source['full_state_digest']):raise AssertionError('Exact model/optimizer/RNG load failed')
        if node['end']==case['total_updates'] and (row['real_targets'],row['valid_anchors'],row['estimated_dense_matrix_ops'])!=(case['end_targets'],case['end_anchors'],case['training_dense_matrix_ops']):raise AssertionError('Cooled source/anchor/work endpoint differs')
        return name,row

    def score(key,capture):
        name,row=capture;point=checkpoint(capture);target=out/'scores'/key;case=plan['cases'][key]
        opts=['--root',str(root),'--plan',str(plan_file),'--plan-sha',execution.sha,'--case',key,'--checkpoint',str(root/'runs'/name/f'checkpoint-u{row["update"]}.pt'),'--checkpoint-sha',point['checkpoint_sha256'],'--cohort','curve','--depth',str(case['primary_inference_depth']),'--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        return execution.cpu(key+'-score','scripts.evaluate_fixed_two_curve_cpu',opts,target/'report.json','COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')

    def reference_score():
        ref=plan['stratification_reference'];target=root/ref['prepared_directory']
        if file_sha256(root/ref['plan'])!=ref['plan_sha256'] or file_sha256(root/ref['checkpoint'])!=ref['checkpoint_sha256']:raise AssertionError('Independent stratification reference changed')
        if file_sha256(target/'report.json')!=ref['prepared_score_sha256'] or file_sha256(root/'fixed-two-strata-reference-v1/report.json')!=ref['parity_report_sha256']:raise AssertionError('Validated independent-reference reader parity changed')
        opts=['--root',str(root),'--plan',str(root/ref['plan']),'--plan-sha',ref['plan_sha256'],'--case',ref['case'],'--checkpoint',str(root/ref['checkpoint']),'--checkpoint-sha',ref['checkpoint_sha256'],'--cohort','curve','--depth','1','--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        return execution.cpu('independent-reference','scripts.evaluate_fixed_two_curve_cpu',opts,target/'report.json','COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')

    def gradient(key,capture):
        name,row=capture;point=checkpoint(capture);phase=IDENTITY+'-'+key+'-gradient'
        common=['--plan-sha',execution.sha,'--case',key,'--checkpoint-sha',point['checkpoint_sha256']]
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',*common,'--checkpoint','/work/instance-nvme/runs/'+name+f'/checkpoint-u{row["update"]}.pt','--comm-port',str(plan['root_comm_ports'][phase])]
        native=execution.native(phase,'scripts.capture_future_span_primitives',opts,'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT')
        target=out/'gradients'/key
        opts=['--root',str(root),'--plan',str(plan_file),*common,'--checkpoint',str(root/'runs'/name/f'checkpoint-u{row["update"]}.pt'),'--capture',str(root/'runs'/phase),'--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        independent=execution.cpu(key+'-gradient','scripts.audit_future_span_gradient',opts,target/'report.json','PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT')
        return dict(native=native,independent_cpu=independent)

    # Independent CPU scoring and audits overlap later native fits. They do
    # not select or cancel an arm because of its measured prediction quality.
    def read_endpoint(key,capture):
        scored=cpu.submit(score,key,capture)
        scored.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
        gate=gradient(key,capture)
        return key,scored.result(),gate

    try:
        reference=cpu.submit(reference_score);reference.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
        write_state(PUBLIC/'status.json',dict(status='RUNNING_FIXED_TWO_NTP_CURVE',maximum_chips=16))
        pending=set(plan['nodes']);running={}
        def priority(node_id):
            node=plan['nodes'][node_id];origin=plan['nodes'][node['parent']]['end'] if node['parent'] else 0
            seconds=plan['measured_budget']['one_warm_seconds_per_update' if plan['cases'][node['case']]['arm']=='one' else 'fixed_warm_seconds_per_update']
            # Stable leaders unlock several independent cooled branches.
            return (node_id.endswith('-stable'),(node['end']-origin)*seconds,node_id)
        with cf.ThreadPoolExecutor(2) as workers:
            while pending or running:
                execution.check()
                ready=sorted((key for key in pending if plan['nodes'][key]['parent'] is None or plan['nodes'][key]['parent'] in captures),key=priority,reverse=True)
                for node_id in ready[:2-len(running)]:
                    parent=plan['nodes'][node_id]['parent'];future=workers.submit(fit,node_id,captures[parent] if parent else None)
                    future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
                    running[future]=node_id;pending.remove(node_id)
                if not running:raise RuntimeError('Finite scientific dependency graph cannot make progress')
                done,_=cf.wait(running,timeout=5,return_when=cf.FIRST_COMPLETED)
                for future in done:
                    node_id=running.pop(future);capture=future.result();captures[node_id]=capture;work_records.append(dict(node=node_id,identity=capture[0],report=capture[1]))
                    write_state(PUBLIC/'training-progress.json',dict(records=work_records))
                    key=plan['nodes'][node_id]['case']
                    if plan['nodes'][node_id]['end']==plan['cases'][key]['total_updates']:
                        diagnostic_handles[key]=diagnostics.submit(read_endpoint,key,capture)
                        diagnostic_handles[key].add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
                for key,future in list(diagnostic_handles.items()):
                    if key not in readouts and future.done():
                        _,readouts[key],gates[key]=future.result()
                        write_state(PUBLIC/'readout-progress.json',dict(scores=readouts,gradient_gates=gates))
        write_state(PUBLIC/'status.json',dict(status='FINISHING_FIXED_CURVE_INDEPENDENT_AUDITS',maximum_chips=16))
        for key,future in diagnostic_handles.items():
            if key not in readouts:_,readouts[key],gates[key]=future.result()
        if set(readouts)!=set(plan['cases']) or set(gates)!=set(plan['cases']):raise AssertionError('Every frozen cooled endpoint and audit must finish')
        actual_updates=sum(x['report']['research_optimizer_updates'] for x in work_records)
        if actual_updates!=plan['planned_actual_research_updates']:raise AssertionError('Actual shared/branched research-update accounting differs')
        documents={}
        for key in readouts:
            rows=json.loads((out/'scores'/key/'document-scores.json').read_text())['records']
            documents[key]=[DocumentScore(v['domain'],v['id'],v['group'],v['nll_sum'],v['real_targets']) for v in rows]
        comparisons={}
        for label in ['1b','2b']:
            for kind,base in [('equal_data','one-'+label),('matched_work','one-work-'+label)]:
                candidate='fixed-'+label
                comparisons[kind+'-'+label]=paired_group_interval(documents[base],documents[candidate],seed=5231,weights=plan['evaluation']['weights'])
        primary=comparisons['matched_work-2b']
        ref_report=reference.result();frames=[];arrays=[]
        for folder,report in [(root/plan['stratification_reference']['prepared_directory'],ref_report),(out/'scores/one-work-2b',readouts['one-work-2b']),(out/'scores/fixed-2b',readouts['fixed-2b'])]:
            if file_sha256(folder/'token-losses.npy')!=report['token_losses_sha256'] or file_sha256(folder/'token-loss-index.json')!=report['token_loss_index_sha256'] or report['evaluation_source_sha256']!=plan['source_sha256']['scripts/evaluate_fixed_two_curve_cpu.py']:raise AssertionError('Private original-position loss or reader changed')
            frames.append(json.loads((folder/'token-loss-index.json').read_text())['records']);arrays.append(np.load(folder/'token-losses.npy',mmap_mode='r'))
        if frames[0]!=frames[1] or frames[0]!=frames[2]:raise AssertionError('Stratification must pair the exact original targets/groups/order')
        strata=reference_deciles(*arrays,frames[0],weights=plan['evaluation']['weights'])
        if abs(strata['pooled_gain_nats']-primary['gain_nats'])>1e-10:raise AssertionError('Secondary bins cannot change the primary result')
        write_state(PUBLIC/'reference-loss-strata.json',strata)
        result=dict(status='COMPLETE_FIXED_TWO_NTP_CURVE',scores=readouts,independent_gradient_gates=gates,comparisons=comparisons,primary='matched_work-2b',primary_mean_clears_useful_threshold=primary['gain_nats']>=.01,actual_research_updates=actual_updates,training_records=work_records,stratification_reference=ref_report,secondary_strata=strata,scope=plan['scope'])
        execution.check()
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),training_records=work_records,scores=readouts,gradient_gates=gates);raise
    finally:
        execution.failed.set();diagnostics.shutdown(wait=True,cancel_futures=True);cpu.shutdown(wait=True,cancel_futures=True)
        execution.close(PUBLIC/'execution-costs.json')
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=execution.sha,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--wait-tune',action='store_true');a=p.parse_args()
    lock=public=None
    if a.wait_tune:
        directory,public,_=queue_plan(a.root)
        if a.prepare_only:return
        lock=(directory/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if a.wait_tune and not (a.root/IDENTITY/'execution/execution-start.json').exists():wait_for_tune(a.root)
        if not (a.root/IDENTITY/'plan.json').exists():register(a.root)
        if not a.prepare_only:
            result=run(a.root,resume=a.resume)
            if public:write_state(public/'report.json',dict(status='TERMINAL_FIXED_CURVE_QUEUE',main_status=result['status'],main_plan_sha256=file_sha256(a.root/IDENTITY/'plan.json'),test_accessed=False,terminal_alpha_spent=0,terminal_success=False,scope='Parent/MAIN receipts carry actual native and CPU cost;this orchestration record adds no reservation interval.'))
    except BaseException as error:
        if public:write_state(public/'report.json',dict(status='STOPPED_FIXED_CURVE_QUEUE',error=str(error),main_admitted=(a.root/IDENTITY/'execution/execution-start.json').exists(),test_accessed=False,terminal_alpha_spent=0,terminal_success=False))
        raise
    finally:
        if lock:lock.close()


if __name__=='__main__':main()
