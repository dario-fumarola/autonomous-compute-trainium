"""Bounded four-cell discovery queue;never a terminal-confirmation launcher."""
import argparse
import concurrent.futures as cf
import json
import os
import signal
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_analysis import point_contrasts,interaction_group_interval,pilot_decision
from compute_trainium.future_span_execution import Execution
from compute_trainium.future_span_study_plan import CELLS,select_rates
from compute_trainium.metrics import DocumentScore,paired_group_interval
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.register_future_span_primitive_study import register


def checkpoint_row(report,update):return next(r for r in report['checkpoint_records'] if r['update']==update)


def wait_for_driver(root):
    """Admission has an original deadline and freezes exact existing paths."""
    out=root/'future-span-primitive-science-queue-v2';public=Path('results/future_span_science_queue_20261025/primitive-v2')
    file=out/'plan.json'
    if file.exists():saved=json.loads(file.read_text())
    else:
        qfile=root/'future-span-primitive-driver-queue-v2/plan.json';queue=json.loads(qfile.read_text())
        prep_file=root/'future-span-study-preparation-v1/manifest.json';prep=json.loads(prep_file.read_text())
        source=dict(queue['source_sha256'])
        for path,sha in prep['source_sha256'].items():
            if path in source and source[path]!=sha:raise AssertionError('Preparation conflicts with existing qualified source')
            source[path]=sha
        for path in ['compute_trainium/future_span_primitive_budget.py','tests/test_future_span_primitive_budget.py',
                     'scripts/capture_future_span_primitives.py','scripts/train_future_span_primitives.py',
                     'compute_trainium/future_span_primitive_driver.py','compute_trainium/future_span_evaluation.py','compute_trainium/future_span_gradient_point.py',
                     'compute_trainium/future_span_study_plan.py','compute_trainium/future_span_analysis.py',
                     'scripts/register_future_span_primitive_study.py','scripts/run_future_span_primitive_study.py',
                     'scripts/evaluate_future_span_cpu.py','scripts/capture_future_span_gradient.py',
                     'scripts/audit_future_span_gradient.py','tests/test_future_span_gradient_point.py',
                     'tests/test_future_span_study_plan.py','tests/test_future_span_primitive_study_admission.py',
                     'tests/test_future_span_analysis.py']:
            sha=file_sha256(path)
            if path in source and source[path]!=sha:raise AssertionError('Scientific queue conflicts with qualified source')
            source[path]=sha
        saved={'identity':'future-span-primitive-science-queue-v2','registered_unix':time.time(),'source_sha256':source,
            'driver_queue_plan_sha256':file_sha256(qfile),'fresh_preparation_manifest_sha256':file_sha256(prep_file),
            'latest_predecessor_finish_unix':queue['latest_predecessor_finish_unix']+3600,
            'assigned_chips':0,'test_accessed':False,'terminal_alpha_spent':0,
            'scope':'Wait for independent complete hardware and driver gates,then register measured-budget TUNE and a separate fixed two-seed main stage. No scientific launch if any prerequisite fails,budget does not fit,source changes or original admission window expires. No terminal confirmation or architecture/horizon/weight search.'}
        out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False);write_state(file,saved);write_state(public/'plan.json',saved)
    target=root/'future-span-primitive-driver-v2/report.json'
    while True:
        if any(file_sha256(p)!=sha for p,sha in saved['source_sha256'].items()):raise AssertionError('Frozen scientific queue source changed')
        if file_sha256(root/'future-span-study-preparation-v1/manifest.json')!=saved['fresh_preparation_manifest_sha256']:
            raise AssertionError('Fresh scientific preparation changed')
        for prerequisite in ['future-span-math-qualification-v2/report.json','future-span-primitive-driver-v2/report.json',
                             'future-span-primitive-driver-queue-v2/report.json']:
            path=root/prerequisite
            if path.exists() and not json.loads(path.read_text())['status'].startswith('PASS_'):
                raise RuntimeError('Prerequisite stopped or failed;do not launch science')
        if target.exists() and not managed_running():
            if json.loads(target.read_text())['status']!='PASS_COMPLETE_FUTURE_SPAN_SCIENTIFIC_DRIVER':
                raise RuntimeError('Complete independent driver pass required')
            write_state(public/'status.json',{'status':'PREREQUISITES_COMPLETE_CHECKING_MEASURED_SCIENTIFIC_BUDGET','assigned_chips':0})
            return
        if time.time()>saved['latest_predecessor_finish_unix']:raise TimeoutError('Original scientific admission window elapsed')
        write_state(public/'status.json',{'status':'WAITING_FOR_COMPLETE_HARDWARE_AND_DRIVER_PASSES','assigned_chips':0})
        time.sleep(10)


def run_stage(root,stage,*,resume=False):
    identity='future-span-'+stage+'-v2';out=root/identity;public=Path('results/future_span_'+stage+'_20261025/primitive-v2')
    plan_file=out/'plan.json'
    if not plan_file.exists():out,public,plan=register(root,stage)
    else:plan=json.loads(plan_file.read_text())
    if (out/'report.json').exists():
        previous=json.loads((out/'report.json').read_text())
        expected='COMPLETE_BOUNDED_FUTURE_SPAN_'+('TUNING' if stage=='tune' else 'OBJECTIVE_PILOT')
        if previous['status']==expected and previous['plan_sha256']==file_sha256(plan_file):return previous
        raise FileExistsError('Preserve a terminal stopped/failed study;explicit recovery identity required')
    execution=Execution(root,plan_file,out/'execution',resume=resume);sha=execution.sha
    cpu_pool=cf.ThreadPoolExecutor(3);fit_records=[];score_records=[];gradient_records=[]
    score_handles={};result={'status':'STOPPED_BOUNDED_FUTURE_SPAN_'+stage.upper()}
    def interrupted(signum,frame):execution.failed.set();raise KeyboardInterrupt('Future study interrupted by signal '+str(signum))
    old_handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def fit(key,phase,end,parent=None):
        case=plan['cases'][key];name=identity+'-'+key+'-'+phase
        options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+identity+'/plan.json',
            '--plan-sha',sha,'--case',key,'--segment-end',str(end),'--comm-port',str(plan['root_comm_ports'][name])]
        if parent:
            origin=parent[1]['update'];row=checkpoint_row(parent[1],origin)
            options+=['--resume','/work/instance-nvme/runs/'+parent[0]+f'/checkpoint-u{origin}.pt',
                '--resume-sha',row['checkpoint_sha256']]
        report=execution.native(name,'scripts.train_future_span_primitives',options,
            'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT')
        if (report['plan_sha256']!=sha or report['case_id']!=key or report['update']!=end or report['qualification_only']
                or report['seed']!=case['seed'] or report['cell']!=case['cell']):raise AssertionError('Scientific fit receipt identity differs')
        for row in report['checkpoint_records']:
            if (file_sha256(root/'runs'/name/f'checkpoint-u{row["update"]}.pt')!=row['checkpoint_sha256']
                    or not row['all_complete_replica_states_equal']):raise AssertionError('Scientific checkpoint or replicas differ')
        if parent:
            if (not report['complete_state_on_load_exact'] or not report['cpu_rng_on_load_exact']
                    or report['recovered_state_digest']!=checkpoint_row(parent[1],parent[1]['update'])['full_state_digest']):
                raise AssertionError('Actual complete optimizer/RNG resume failed')
        if end==case['total_updates']:
            if (report['real_targets']!=case['end_targets'] or report['valid_anchors']!=case['end_anchors']
                    or report['estimated_dense_matrix_ops']!=case['training_dense_matrix_ops']):
                raise AssertionError('Completed endpoint natural/auxiliary/work counters differ')
        with execution.lock:
            fit_records.append({'identity':name,**{k:v for k,v in report.items() if k not in ['progress','first_global_probe']}})
            write_state(public/'training-progress.json',{'records':fit_records})
        return name,report
    def score(key,capture,update,depth,label):
        name,report=capture;row=checkpoint_row(report,update);case=plan['cases'][key]
        target=out/'scores'/f'{key}-{label}-d{depth}';cohort='tune' if stage=='tune' else 'curve'
        options=['--root',str(root),'--plan',str(plan_file),'--plan-sha',sha,'--checkpoint',
            str(root/'runs'/name/f'checkpoint-u{update}.pt'),'--checkpoint-sha',row['checkpoint_sha256'],
            '--case',key,'--cohort',cohort,'--depth',str(depth),'--output',str(target),
            '--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        scored=execution.cpu(key+'-'+label+'-d'+str(depth),'scripts.evaluate_future_span_cpu',options,
            target/'report.json','COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')
        if scored['checkpoint_sha256']!=row['checkpoint_sha256']:raise AssertionError('Independent score checkpoint differs')
        scored.update(peak_lr=case['peak_lr'],readout=label)
        with execution.lock:score_records.append(scored);write_state(public/'scores-progress.json',{'records':score_records})
        return scored
    def gradient(key,capture):
        name,report=capture;update=report['update'];row=checkpoint_row(report,update)
        phase=identity+'-'+key+'-gradient';checkpoint=root/'runs'/name/f'checkpoint-u{update}.pt'
        common=['--plan-sha',sha,'--case',key,'--checkpoint-sha',row['checkpoint_sha256']]
        options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+identity+'/plan.json',*common,
            '--checkpoint','/work/instance-nvme/runs/'+name+f'/checkpoint-u{update}.pt',
            '--comm-port',str(plan['root_comm_ports'][phase])]
        native=execution.native(phase,'scripts.capture_future_span_primitives',options,'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT')
        if native['checkpoint_sha256']!=row['checkpoint_sha256'] or native['plan_sha256']!=sha:
            raise AssertionError('Captured fitted-gradient point differs')
        target=out/'gradients'/key
        options=['--root',str(root),'--plan',str(plan_file),*common,'--checkpoint',str(checkpoint),
            '--capture',str(root/'runs'/phase),'--output',str(target),
            '--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        audited=cpu_pool.submit(execution.cpu,key+'-gradient-audit','scripts.audit_future_span_gradient',options,
            target/'report.json','PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT').result()
        with execution.lock:
            gradient_records.append({'case_id':key,'identity':phase,'native':native,'independent_cpu':audited})
            write_state(public/'gradient-progress.json',{'records':gradient_records})
        return audited
    def worker(key):
        try:
            case=plan['cases'][key]
            if stage=='tune':return fit(key,'fit',576)
            first=fit(key,'first',960)
            futures=[cpu_pool.submit(score,key,first,576,case['primary_inference_depth'],'stable301m')]
            end=fit(key,'finish',1920,first);gradient(key,end)
            for depth in [1] if case['arm']=='one' else [1,2,3]:
                futures.append(cpu_pool.submit(score,key,end,1920,depth,'cooled1b'))
            for future in futures:future.result(timeout=max(1,execution.deadline-time.time()))
            return end
        except BaseException:execution.failed.set();raise
    try:
        for folder,receipt in plan['launch_receipts'].items():
            if file_sha256(root/folder/'report.json')!=receipt['report_sha256']:raise AssertionError('Frozen launch prerequisite changed')
        captures={}
        with cf.ThreadPoolExecutor(2) as pool:
            jobs={key:pool.submit(worker,key) for key in plan['cases']}
            for key,future in jobs.items():
                captures[key]=future.result()
                if stage=='tune':
                    depth=plan['cases'][key]['primary_inference_depth']
                    score_handles[key]=cpu_pool.submit(score,key,captures[key],576,depth,'cooled301m')
        if stage=='tune':
            selected=select_rates([future.result(timeout=max(1,execution.deadline-time.time())) for future in score_handles.values()])
            write_state(public/'selection.json',{'selected':selected,'scope':'Frozen grid;boundary winners remain unbracketed;no extra rates.'})
            with cf.ThreadPoolExecutor(2) as pool:
                jobs=[pool.submit(gradient,row['case_id'],captures[row['case_id']]) for row in selected.values()]
                for future in jobs:future.result(timeout=max(1,execution.deadline-time.time()))
            result={'status':'COMPLETE_BOUNDED_FUTURE_SPAN_TUNING','selected':selected,
                'selected_gradient_gates':gradient_records,'tuning_scores':score_records}
        else:
            per_seed={};groups={}
            for seed in [5231,7283]:
                rows=[r for r in score_records if r['seed']==seed and r['readout']=='cooled1b']
                values={(r['cell'],r['depth']):r['metrics']['weighted_nll'] for r in rows}
                if len(rows)!=8 or len(values)!=8:raise AssertionError('All frozen final exits required without duplication')
                per_seed[seed]=point_contrasts(values);docs={}
                for r in rows:
                    path=out/'scores'/f'{r["case_id"]}-cooled1b-d{r["depth"]}'/'document-scores.json'
                    docs[(r['cell'],r['depth'])]=[DocumentScore(v['domain'],v['id'],v['group'],v['nll_sum'],v['real_targets'])
                        for v in json.loads(path.read_text())['records']]
                primary={cell:docs[(cell,1 if cell.startswith('one') else 2)] for cell in CELLS}
                groups[seed]={'interaction':interaction_group_interval(primary,seed=seed),
                    'recurrent_future_over_one_future':paired_group_interval(primary['one-future'],primary['recurrent-future'],seed=seed),
                    'recurrent_future_over_recurrent_ntp':paired_group_interval(primary['recurrent-ntp'],primary['recurrent-future'],seed=seed),
                    'recurrent_future_depth1_to2':paired_group_interval(docs[('recurrent-future',1)],primary['recurrent-future'],seed=seed)}
            result={'status':'COMPLETE_BOUNDED_FUTURE_SPAN_OBJECTIVE_PILOT','paired_seed_point_contrasts':per_seed,
                'conditional_source_group_intervals':groups,'decision':pilot_decision(per_seed),
                'independent_gradient_gates':gradient_records,'scores':score_records}
        execution.check()
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),training_records=fit_records,scores=score_records,gradient_gates=gradient_records);raise
    finally:
        execution.failed.set();cpu_pool.shutdown(wait=True,cancel_futures=True)
        execution.close(public/'execution-costs.json')
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
        result.update(plan_sha256=sha,test_accessed=False,terminal_alpha_spent=0,terminal_success=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['tune','main'],default='tune');p.add_argument('--resume',action='store_true')
    p.add_argument('--wait-driver',action='store_true');p.add_argument('--chain-main',action='store_true');a=p.parse_args()
    if a.wait_driver:
        try:wait_for_driver(a.root)
        except BaseException as error:
            target=Path('results/future_span_science_queue_20261025/primitive-v2')
            if target.exists():write_state(target/'report.json',{'status':'STOPPED_BEFORE_FUTURE_SCIENTIFIC_ADMISSION',
                'error':str(error),'assigned_chips':0,'research_optimizer_updates':0,'test_accessed':False})
            raise
    result=run_stage(a.root,a.stage,resume=a.resume)
    if a.chain_main and a.stage=='tune':
        if result['status']!='COMPLETE_BOUNDED_FUTURE_SPAN_TUNING':raise RuntimeError('All tuning stages/gates required')
        run_stage(a.root,'main')


if __name__=='__main__':main()
