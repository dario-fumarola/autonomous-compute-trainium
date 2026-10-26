"""Bounded three-rate TUNE for the frozen final-quarter depth schedule."""
import argparse
import concurrent.futures as cf
import copy
import json
import os
from pathlib import Path
import signal
import time
import traceback

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_fixed_two_ntp_tune import GuardedExecution
from scripts.run_fixed_control_replication import reserves

IDENTITY='late-two-allocation-tune-v1'
PREP='late-two-allocation-preparation-v1'
PUBLIC=Path('results/late_two_allocation_20261026/tune')
DOC='docs/LATE_TWO_ALLOCATION_20261026.md'
ADMISSION_CUTOFF=1793057400  # 2026-10-26 23:30:00 UTC


def read(path):return json.loads(Path(path).read_text())


def measured_warm_seconds(report, *, start, end, depth_table, expected_depth):
    if end-start!=100 or not 0<start<end:
        raise ValueError('Exactly one hundred actual update intervals required')
    if not (depth_table[start:end]==expected_depth).all():
        raise ValueError('A warm measurement may not cross a training-depth switch')
    if any(start<row['update']<end for row in report['checkpoint_records']):
        raise ValueError('Warm interval includes checkpoint work')
    points={p['update']:p for p in report['progress']}
    result=points[end]['elapsed_seconds']-points[start]['elapsed_seconds']
    if result<=0:raise ValueError('Warm elapsed time did not advance')
    return result


def register(root):
    from scripts.register_future_span_primitive_study import prerequisites
    if time.time()>ADMISSION_CUTOFF:raise TimeoutError('Original fixed TUNE admission cutoff elapsed')
    if managed_running():raise RuntimeError('Predecessor native owners must be terminal')
    reserves(root)
    storage_file=root/'closed-source-inode-sharing-v1/report.json'
    storage=read(storage_file)
    if (storage['status']!='COMPLETE_BYTE_PRESERVING_CLOSED_SOURCE_SHARING'
            or not storage['all_launch_and_closure_receipts_unchanged']):
        raise ValueError('Complete source-path/byte/receipt preservation must precede scientific admission')
    if __import__('shutil').disk_usage(root).free<75*1024**3:
        raise RuntimeError('Forty GiB complete TUNE artifacts plus protected storage required')
    inode=project_snapshot_inodes(Path.cwd(),13)
    if os.statvfs(root).f_favail<inode['required_free_inodes']:
        raise RuntimeError('Complete13-phase source/artifact inode projection must fit')
    prepared_file=root/PREP/'manifest.json';prepared=read(prepared_file);prep_receipt=read(root/PREP/'report.json')
    if (prepared['status']!='COMPLETE_CPU_ONLY_LATE_TWO_ALLOCATION_PREPARATION'
            or prep_receipt['manifest_sha256']!=file_sha256(prepared_file)):
        raise ValueError('Complete exact CPU schedule and original-state preparation required')
    registration=read(root/PREP/'registration.json')
    if any(file_sha256(f)!=h for f,h in registration['source_sha256'].items()):
        raise AssertionError('Prepared final-quarter schedule changed')
    qualified,receipts,_=prerequisites(root)
    old_file=root/'future-span-tune-v2/plan.json';old=read(old_file)
    old_fixed=read(root/'fixed-two-ntp-tune-v1/report.json')
    measurements=read(root/'fixed-two-ntp-main-v1/plan.json')['measured_budget']
    one=measurements['one_warm_seconds_per_update']
    two=max(v['warm_seconds']/100 for v in old_fixed['warm'].values());cold=max(250.,measurements['cold_allowance'])
    projected=3*(432*one+144*two)*1.5/2+13*cold/2+4*900/3+360+600
    if projected>7200:raise RuntimeError('Complete measured TUNE including all audits exceeds cap')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve stopped/completed TUNE identities')
    sources={**qualified,**old['source_sha256'],**registration['source_sha256']}
    for f in ['scripts/run_late_two_tune.py','tests/test_late_two_tune.py']:
        sources[f]=file_sha256(f)
    if any(file_sha256(f)!=h for f,h in sources.items()):raise AssertionError('Qualified source stack changed')
    worker,host=split_snapshot_sources(sources)
    endpoint=prepared['endpoints']['tune'];cases={}
    for key,original in old['cases'].items():
        if original['cell']!='recurrent-ntp':continue
        case=copy.deepcopy(original)
        case.update(total_updates=576,end_targets=endpoint['real_targets'],end_anchors=endpoint['end_anchors'],
            cooldown_start_update=288,cooldown_start_targets=endpoint['cooldown_start_targets'],
            depth_table='../'+PREP+'/'+endpoint['depth_file'],depth_table_sha256=endpoint['depth_sha256'],
            work_table='../'+PREP+'/'+endpoint['work_file'],work_table_sha256=endpoint['work_sha256'],
            training_dense_matrix_ops=endpoint['training_matrix_ops'],checkpoint_updates=[121,432,576],
            segment_ends=[121,432,576],gradient_updates=[121,576],primary_inference_depth=2,
            switch_after_update=432)
        cases[key]=case
    if len(cases)!=3 or {c['peak_lr'] for c in cases.values()}!={.008,.016,.032}:
        raise AssertionError('Exactly three original rate opportunities required')
    initial=copy.deepcopy(prepared['initial_states']['s4127-family']);initial['file']='../'+initial['file']
    initial.pop('file_is_root_relative',None)  # engine paths are relative to this plan directory
    names=[IDENTITY+'-'+key+'-'+phase for key in cases for phase in ['prefix','prefix-gradient','one-finish','two-finish','final-gradient']]
    plan=copy.deepcopy(old)
    for key in ['primary_interaction','absolute_candidate_gates','primary_interval_scope','strongest_baseline_required']:
        plan.pop(key,None)
    plan.update(identity=IDENTITY,registered_unix=time.time(),cases=cases,
        initial_states={'s4127-family':initial},auxiliary_data=prepared['auxiliary_data'],
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),seconds_limit=7200,phase_seconds_limit=1200,cpu_phase_seconds_limit=1200,
        admission_cutoff_unix=ADMISSION_CUTOFF,composite_seconds_limit=21600,
        preparation_manifest_sha256=file_sha256(prepared_file),launch_receipts=receipts,
        completed_source_storage_receipt_sha256=file_sha256(storage_file),
        historical_ordinary8_tune_report_sha256=file_sha256(root/'lean-baseline-tune-v1/report.json'),
        measured_budget=dict(one_seconds_per_update=one,two_seconds_per_update=two,cold_allowance=cold,
            complete_seconds_with_margin=projected,cap_seconds=7200),
        complete_source_inode_projection=inode,primary_scope='TUNE selection only;no scientific quality or efficiency claim',
        test_accessed=False,terminal_alpha_spent=0,scope=Path(DOC).read_text())
    out.mkdir();PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan


def run(root):
    import numpy as np
    out=root/IDENTITY;plan_file=out/'plan.json';plan=read(plan_file)
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal TUNE result;explicit recovery registration required')
    if time.time()>plan['admission_cutoff_unix']:raise TimeoutError('Original fixed admission cutoff elapsed')
    if file_sha256(root/PREP/'manifest.json')!=plan['preparation_manifest_sha256']:
        raise AssertionError('Frozen schedule preparation changed')
    if file_sha256(root/'closed-source-inode-sharing-v1/report.json')!=plan['completed_source_storage_receipt_sha256']:
        raise AssertionError('Verified source-preservation receipt changed')
    execution=GuardedExecution(root,plan_file,out/'execution');cpu=cf.ThreadPoolExecutor(3)
    result=dict(status='STOPPED_LATE_TWO_ALLOCATION_TUNING');prefixes={};before_switch={};fits={};scores={};gates={};warm={}
    def interrupted(number,frame):execution.failed.set();raise KeyboardInterrupt('Late-depth TUNE interrupted: '+str(number))
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    def point(capture):return next(p for p in capture[1]['checkpoint_records'] if p['update']==capture[1]['update'])
    def fit(key,phase,end,parent=None):
        name=IDENTITY+'-'+key+'-'+phase
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
            '--plan-sha',execution.sha,'--case',key,'--segment-end',str(end),'--comm-port',str(plan['root_comm_ports'][name])]
        if parent:
            saved=point(parent);opts+=['--resume','/work/instance-nvme/runs/'+parent[0]+f'/checkpoint-u{saved["update"]}.pt','--resume-sha',saved['checkpoint_sha256']]
        row=execution.native(name,'scripts.train_future_span_primitives',opts,'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT')
        if row['plan_sha256']!=execution.sha or row['case_id']!=key or row['update']!=end or row['qualification_only'] or row['objective_weight']!=0.:
            raise AssertionError('Registered schedule/model/objective changed')
        if any(not p['all_complete_replica_states_equal'] for p in row['checkpoint_records']):raise AssertionError('Saved training replicas differ')
        if parent and (not row['complete_state_on_load_exact'] or not row['cpu_rng_on_load_exact'] or row['recovered_state_digest']!=saved['full_state_digest']):
            raise AssertionError('Complete checkpoint/optimizer/RNG restoration differs')
        if end==576:
            case=plan['cases'][key]
            if (row['real_targets'],row['valid_anchors'],row['estimated_dense_matrix_ops'])!=(case['end_targets'],case['end_anchors'],case['training_dense_matrix_ops']):
                raise AssertionError('Cooled source/head/work endpoint differs')
        return name,row
    def gradient(key,capture,phase):
        name,row=capture;saved=point(capture);native_name=IDENTITY+'-'+key+'-'+phase
        common=['--plan-sha',execution.sha,'--case',key,'--checkpoint-sha',saved['checkpoint_sha256']]
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',*common,
            '--checkpoint','/work/instance-nvme/runs/'+name+f'/checkpoint-u{row["update"]}.pt',
            '--comm-port',str(plan['root_comm_ports'][native_name])]
        native=execution.native(native_name,'scripts.capture_future_span_primitives',opts,'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT')
        target=out/'gradients'/(key+'-'+phase)
        opts=['--root',str(root),'--plan',str(plan_file),*common,'--checkpoint',str(root/'runs'/name/f'checkpoint-u{row["update"]}.pt'),
            '--capture',str(root/'runs'/native_name),'--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        independent=execution.cpu(key+'-'+phase,'scripts.audit_future_span_gradient',opts,target/'report.json','PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT')
        return dict(native=native,independent_cpu=independent)
    def score(key,capture):
        name,row=capture;saved=point(capture);target=out/'scores'/key
        opts=['--root',str(root),'--plan',str(plan_file),'--plan-sha',execution.sha,'--checkpoint',str(root/'runs'/name/'checkpoint-u576.pt'),
            '--checkpoint-sha',saved['checkpoint_sha256'],'--case',key,'--cohort','tune','--depth','2',
            '--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        return execution.cpu(key+'-score','scripts.evaluate_future_span_cpu',opts,target/'report.json','COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')
    def parallel(function,items,workers=2):
        with cf.ThreadPoolExecutor(workers) as pool:
            jobs={k:pool.submit(function,k,v) for k,v in items.items()}
            for future in jobs.values():future.add_done_callback(lambda f:execution.failed.set() if f.exception() else None)
            return {k:future.result() for k,future in jobs.items()}
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_COUNTED_ONE_PASS_PREFIXES',maximum_chips=16))
        def prefix_and_gate(key,unused):
            capture=fit(key,'prefix',121)
            table=np.load(plan_file.parent/plan['cases'][key]['depth_table'],mmap_mode='r')
            timing=dict(one_pass_updates=100,one_pass_seconds=measured_warm_seconds(capture[1],start=20,end=120,depth_table=table,expected_depth=1))
            return capture,gradient(key,capture,'prefix-gradient'),timing
        ready=parallel(prefix_and_gate,plan['cases'],3)
        prefixes={k:v[0] for k,v in ready.items()};gates={k:v[1] for k,v in ready.items()};warm={k:v[2] for k,v in ready.items()}
        # Only timing/numerical gates are read before completing all rate fits.
        one=max(v['one_pass_seconds']/100 for v in warm.values());two=plan['measured_budget']['two_seconds_per_update'];cold=plan['measured_budget']['cold_allowance']
        predicted=3*((432-121)*one+144*two)*1.5/2+7*cold/2+1200+360+600
        readiness=dict(predicted_remaining_seconds_with_margin=predicted,seconds_remaining=execution.deadline-time.time(),actual_warm=warm)
        write_state(PUBLIC/'readiness.json',readiness)
        if predicted>readiness['seconds_remaining']:raise RuntimeError('Complete remaining TUNE does not fit original cap')
        write_state(PUBLIC/'status.json',dict(status='RUNNING_TO_FROZEN_SWITCH_AND_COOLED_TWO_PASS_FINISHES',maximum_chips=16))
        def finish_both_stages(key,value):
            switched=fit(key,'one-finish',432,value)
            return switched,fit(key,'two-finish',576,switched)
        finished=parallel(finish_both_stages,prefixes,3)
        before_switch={k:v[0] for k,v in finished.items()};fits={k:v[1] for k,v in finished.items()}
        for key,capture in fits.items():
            table=np.load(plan_file.parent/plan['cases'][key]['depth_table'],mmap_mode='r')
            warm[key].update(two_pass_updates=100,two_pass_seconds=measured_warm_seconds(capture[1],start=440,end=540,depth_table=table,expected_depth=2))
        score_jobs={key:cpu.submit(score,key,value) for key,value in fits.items()}
        for future in score_jobs.values():future.add_done_callback(lambda f:execution.failed.set() if f.exception() else None)
        scores={key:job.result() for key,job in score_jobs.items()}
        selected=min(scores,key=lambda k:(scores[k]['metrics']['weighted_nll'],plan['cases'][k]['peak_lr']))
        final=gradient(selected,fits[selected],'final-gradient')
        result=dict(status='COMPLETE_QUALIFIED_LATE_TWO_ALLOCATION_TUNING',selected_case=selected,
            selected_rate=plan['cases'][selected]['peak_lr'],boundary_winner=plan['cases'][selected]['peak_lr'] in [.008,.032],
            scores=scores,prefix_gates=gates,selected_final_gate=final,actual_warm=warm,
            completed_fit_reports={k:v[1] for k,v in fits.items()},
            switch_readback_checks={k:dict(update=v[1]['update'],restored_complete_state=fits[k][1]['complete_state_on_load_exact'],
                restored_cpu_rng=fits[k][1]['cpu_rng_on_load_exact'],unchanged_endpoint_schedule=True) for k,v in before_switch.items()},
            scientific_optimizer_updates=3*576,main_scientific_fit_admitted=False)
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),error_kind=type(error).__name__,traceback=traceback.format_exc(),
            prefix_reports={k:v[1] for k,v in prefixes.items()},completed_fit_reports={k:v[1] for k,v in fits.items()},scores=scores,prefix_gates=gates)
        write_state(out/'initiating-failure.json',result);raise
    finally:
        execution.failed.set();cpu.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=execution.sha,scope=plan['scope'],test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if not (a.root/IDENTITY/'plan.json').exists():register(a.root)
    if not a.prepare_only:run(a.root)


if __name__=='__main__':main()
