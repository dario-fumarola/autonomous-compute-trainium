"""Separately admitted,work-matched paired late-depth development comparison."""
import argparse
import concurrent.futures as cf
import copy
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import threading
import time
import traceback

import numpy as np

from compute_trainium.execution_isolation import managed_running
from compute_trainium.fixed_two_curve import stop_on_failed_future
from compute_trainium.future_span_gradient_point import point_tables,validate_point
from compute_trainium.future_span_scientific_state import signature
from compute_trainium.late_allocation_main import (SEEDS,WEIGHTS,bounded_deadline,
    qualified_rate,gradient_gate,seed_summary,validate_rng_inventory,final_score_gate)
from compute_trainium.metrics import DocumentScore,paired_group_interval,summarize
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.qualify_native import digest
from scripts.run_fixed_control_replication import reserves
from scripts.run_fixed_two_ntp_tune import GuardedExecution

IDENTITY='late-two-allocation-main-v1'
TUNE='late-two-allocation-tune-v1'
PREP='late-two-allocation-preparation-v1'
PUBLIC=Path('results/late_two_allocation_20261026/main')
DOC='docs/LATE_TWO_MAIN_ADMISSION_20261026.md'


def read(path):return json.loads(Path(path).read_text())


def complete_sources(parent):
    # Pin all in-repository executable code,including host admission helpers.
    sources=dict(parent)
    for tree in ['compute_trainium','scripts','configs','third_party']:
        for path in Path(tree).rglob('*.py'):
            if path.is_file() and not path.is_symlink():sources[str(path)]=file_sha256(path)
    for path in [DOC,'tests/test_late_allocation_main.py']:
        sources[path]=file_sha256(path)
    return sources


def fixed_reference(root,seed,evaluation):
    directory='fixed-two-ntp-main-v1' if seed==5231 else 'fixed-control-replication-main-v1'
    key='fixed-2b' if seed==5231 else 'fixed-s17293-data'
    file=root/directory/'scores'/key/'report.json';score=read(file);parent=root/directory/'report.json'
    gates=read(parent)['independent_gradient_gates' if seed==5231 else 'gradient_gates']
    cpu=gates[key]['independent_cpu'];documents=file.parent/'document-scores.json'
    gate=gates[key];native_binding={}
    if 'native' not in gate:
        native_file=root/'runs'/(directory+'-'+key+'-gradient')/'report.json'
        if file_sha256(native_file)!=gate['native_report_sha256']:
            raise AssertionError('Historical separately stored native audit changed')
        gate=dict(native=read(native_file),independent_cpu=cpu)
        native_binding=dict(native_report_file=str(native_file.relative_to(root)),
            native_report_sha256=file_sha256(native_file))
    if (score['seed']!=seed or score['depth']!=2 or score['training_update']!=3840
            or score['source_corpus_sha256']!=evaluation['corpus_sha256']
            or cpu['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
            or cpu['checkpoint_sha256']!=score['checkpoint_sha256']):
        raise AssertionError('Exact original audited fixed-two reference required')
    gradient_gate(gate,key,3840,score['checkpoint_sha256'])
    return dict(file=str(file.relative_to(root)),sha256=file_sha256(file),
        parent_file=str(parent.relative_to(root)),parent_report_sha256=file_sha256(parent),document_scores=str(documents.relative_to(root)),
        document_scores_sha256=file_sha256(documents),checkpoint_sha256=score['checkpoint_sha256'],
        weighted_nll=score['metrics']['weighted_nll'],**native_binding)


def ordinary_reference(root,seed,evaluation,budget):
    parent=root/'lean-baseline-main-ordinary8-v1/report.json';report=read(parent);key=str(seed)
    score=report['scores'][key];gate=report['final_gradient_gates'][key];cpu=gate['independent_cpu'];native=gate['native']
    folder=parent.parent/'scores'/key
    if (report['status']!='COMPLETE_QUALIFIED_LEAN_BASELINE_MAIN_CONTROL'
            or score!=read(folder/'report.json') or score['seed']!=seed or score['depth']!=1
            or score['parameters']!=141733120 or score['training_update']!=5275
            or score['source_corpus_sha256']!=evaluation['corpus_sha256']
            or score['metrics']['weights']!=WEIGHTS or not score['full_state_readback_exact']
            or cpu['status']!='PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT'
            or not cpu['gradient']['passes_two_percent_gate'] or not cpu['full_state_readback_exact']
            or cpu['checkpoint_sha256']!=score['checkpoint_sha256'] or not native['replica_gradients_equal']
            or cpu['windows']!=512 or native['windows']!=512
            or not 0<budget-score['training_executed_matmuls']<budget/1000):
        raise AssertionError('Exact independently tuned,audited same-budget ordinary8 reference required')
    document=folder/'document-scores.json'
    return dict(file=str((folder/'report.json').relative_to(root)),sha256=file_sha256(folder/'report.json'),
        parent_file=str(parent.relative_to(root)),parent_report_sha256=file_sha256(parent),document_scores=str(document.relative_to(root)),
        document_scores_sha256=file_sha256(document),checkpoint_sha256=score['checkpoint_sha256'],
        weighted_nll=score['metrics']['weighted_nll'],training_matrix_ops=score['training_executed_matmuls'])


def measured_budget(tune):
    one=max(v['one_pass_seconds']/100 for v in tune['actual_warm'].values())
    two=max(v['two_pass_seconds']/100 for v in tune['actual_warm'].values())
    audit=max([g['independent_cpu']['wall_seconds'] for g in tune['prefix_gates'].values()]
              +[tune['selected_final_gate']['independent_cpu']['wall_seconds']])
    # Both seeds run concurrently;all six complete audits and both final
    # independent readouts are included conservatively rather than assumed free.
    parts=dict(training=(3300*one+1100*two)*1.5,cold_native_phases=12*250/2,
        six_independent_full_gradients=6*max(600.,audit*1.5)/2,
        independent_scores=600,closure_margin=300)
    return dict(one_seconds_per_update=one,two_seconds_per_update=two,
        largest_observed_cpu_audit_seconds=audit,parts=parts,complete_seconds=sum(parts.values()))


def register(root):
    if managed_running():raise RuntimeError('TUNE and all previous native owners must be terminal')
    tune_file=root/TUNE/'report.json';tune=read(tune_file);tune_plan_file=root/TUNE/'plan.json';old=read(tune_plan_file)
    if tune['plan_sha256']!=file_sha256(tune_plan_file):raise AssertionError('Complete exact TUNE plan identity required')
    rate=qualified_rate(tune,old)
    if any(file_sha256(f)!=h for f,h in old['source_sha256'].items()):raise AssertionError('Qualified TUNE executable source changed')
    start=read(root/TUNE/'execution/execution-start.json');original_deadline=start['started_unix']+21600
    if original_deadline!=read(root/TUNE/'composite-clock.json')['original_composite_deadline_unix']:
        raise AssertionError('Original composite deadline changed')
    now=time.time();available=bounded_deadline(now,original_deadline)-now
    estimate=measured_budget(tune)
    if estimate['complete_seconds']+120>available:raise RuntimeError('Complete paired MAIN cannot fit original composite/local caps')
    reserves(root)
    if shutil.disk_usage(root).free<75*1024**3:raise RuntimeError('Complete projected MAIN artifacts plus35GiB reserve required')
    inodes=project_snapshot_inodes(Path.cwd(),12)
    if os.statvfs(root).f_favail<inodes['required_free_inodes']:raise RuntimeError('All12 native snapshots/artifacts plus inode reserve required')
    prepared_file=root/PREP/'manifest.json';prepared=read(prepared_file)
    if (prepared['status']!='COMPLETE_CPU_ONLY_LATE_TWO_ALLOCATION_PREPARATION'
            or read(root/PREP/'report.json')['manifest_sha256']!=file_sha256(prepared_file)):
        raise AssertionError('Original exact schedule/state preparation required')
    registration=read(root/PREP/'registration.json')
    if any(file_sha256(f)!=h for f,h in registration['source_sha256'].items()):raise AssertionError('Prepared final-quarter schedule changed')
    endpoint=prepared['endpoints']['main'];budget=prepared['main_work_floor']['budget_matrix_ops']
    if (endpoint['total_updates']!=4400 or endpoint['switch_after_update']!=3300
            or endpoint['real_targets']!=2302851264 or budget!=1272689958300549120):
        raise AssertionError('No budget/exposure/switch search or4B extension is admitted')
    references={str(s):dict(ordinary8=ordinary_reference(root,s,prepared['evaluation'],budget),
        fixed_two=fixed_reference(root,s,prepared['evaluation'])) for s in SEEDS}
    # Independently bind all three switch reports/actual optimizer-call counts.
    for key in old['cases']:
        previous=None
        for phase,left,right in [('prefix',0,121),('one-finish',121,432),('two-finish',432,576)]:
            file=root/'runs'/(TUNE+'-'+key+'-'+phase)/'report.json';row=read(file)
            if row['actual_optimizer_calls_per_replica']!=right-left or row['update']!=right:
                raise AssertionError('TUNE phase hid or omitted actual updates')
            point=next(p for p in row['checkpoint_records'] if p['update']==right)
            checkpoint=file.parent/f'checkpoint-u{right}.pt'
            if file_sha256(checkpoint)!=point['checkpoint_sha256']:raise AssertionError('TUNE phase checkpoint bytes changed')
            if previous and (row['recovered_state_digest']!=previous['full_state_digest']
                    or not row['complete_state_on_load_exact'] or not row['cpu_rng_on_load_exact']):
                raise AssertionError('Exact TUNE prefix/switch state replay required')
            previous=point
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every MAIN registration and result')
    sources=complete_sources(old['source_sha256']);worker,host=split_snapshot_sources(sources)
    initials={};cases={}
    for seed in SEEDS:
        initial_key=f's{seed}-family';info=copy.deepcopy(prepared['initial_states'][initial_key])
        if file_sha256(root/info['file'])!=info['sha256']:raise AssertionError('Original paired random initializer changed')
        info['file']='../'+info['file'];info.pop('file_is_root_relative',None);initials[initial_key]=info
        key=f'late-two-ntp-s{seed}'
        cases[key]=dict(cell='recurrent-ntp',arm='recurrent',objective_weight=0.,seed=seed,state_seed=seed+2009,
            initial_state=initial_key,training_parameters=166571840,peak_lr=rate,warmup_targets=50000000,
            cooldown_start_update=2200,cooldown_start_targets=endpoint['cooldown_start_targets'],
            total_updates=4400,end_targets=endpoint['real_targets'],end_anchors=endpoint['end_anchors'],
            training_dense_matrix_ops=endpoint['training_matrix_ops'],depth_table='../'+PREP+'/'+endpoint['depth_file'],
            depth_table_sha256=endpoint['depth_sha256'],work_table='../'+PREP+'/'+endpoint['work_file'],
            work_table_sha256=endpoint['work_sha256'],checkpoint_updates=[121,3300,4400],
            segment_ends=[121,3300,4400],gradient_updates=[121,3300,4400],primary_inference_depth=2,switch_after_update=3300)
    names=[IDENTITY+'-'+key+'-'+phase for key in cases for phase in ['prefix','prefix-gradient','one-finish','switch-gradient','two-finish','final-gradient']]
    plan=copy.deepcopy(old)
    plan.update(identity=IDENTITY,registered_unix=now,stage='main',cases=cases,initial_states=initials,
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),seconds_limit=14400,phase_seconds_limit=7200,cpu_phase_seconds_limit=1200,
        original_composite_deadline_unix=original_deadline,original_tune_started_unix=start['started_unix'],
        measured_budget=estimate,complete_source_inode_projection=inodes,tune_report_sha256=file_sha256(tune_file),
        tune_plan_sha256=file_sha256(tune_plan_file),preparation_manifest_sha256=file_sha256(prepared_file),
        references=references,selected_rate=rate,minimum_useful_gain_nats=.01,
        source_total_real_targets=6500009163,primary_comparator='ordinary8',primary_scope='Paired two-seed consumed-DEV development screen only',
        test_accessed=False,terminal_alpha_spent=0,scope=Path(DOC).read_text())
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan


class MainExecution(GuardedExecution):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        relative=self.deadline
        self.deadline=bounded_deadline(self.execution['started_unix'],self.plan['original_composite_deadline_unix'])
        self.execution.update(deadline_unix=self.deadline,relative_main_deadline_unix=relative,
            original_composite_deadline_unix=self.plan['original_composite_deadline_unix'])
        write_state(self.out/'execution-start.json',self.execution)
    def check(self):
        super().check()
        if time.time()>self.plan['original_composite_deadline_unix']:
            raise TimeoutError('Original composite deadline;no renewal')


def checkpoint(capture):
    return next(p for p in capture[1]['checkpoint_records'] if p['update']==capture[1]['update'])


def verify_readback(root,plan_file,key,capture):
    import torch
    started=time.time();plan=read(plan_file);case=plan['cases'][key];name,row=capture;point=checkpoint(capture)
    file=root/'runs'/name/f'checkpoint-u{row["update"]}.pt'
    if file_sha256(file)!=point['checkpoint_sha256']:raise AssertionError('Counted checkpoint bytes changed')
    saved=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
    counts,anchors,depths,work=point_tables(root,plan_file,plan,case);cursor=saved['cursor']
    validate_point(saved,file_sha256(plan_file),plan,key,counts[cursor]-counts[0],anchors[cursor],work[:saved['update']].sum())
    validate_rng_inventory(saved['cpu_rng_by_rank'])
    if (digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})!=saved['model_optimizer_digest']
            or saved['model_optimizer_digest']!=point['full_state_digest'] or not point['all_complete_replica_states_equal']):
        raise AssertionError('Exact complete model/optimizer readback or replicas differ')
    return dict(update=row['update'],checkpoint_sha256=point['checkpoint_sha256'],complete_state_readback_exact=True,
        all32_rng_states_recoverable=True,replicas_equal=True,next_forward_backward_depth=list(map(int,depths[saved['update'],0,0])),
        wall_seconds=time.time()-started)


def compare(root,reference,directory):
    file=root/reference['file'];docs=root/reference['document_scores']
    if (file_sha256(file)!=reference['sha256'] or file_sha256(docs)!=reference['document_scores_sha256']
            or file_sha256(root/reference['parent_file'])!=reference['parent_report_sha256']):
        raise AssertionError('Frozen paired ordinary/fixed reference changed')
    if ('native_report_file' in reference
            and file_sha256(root/reference['native_report_file'])!=reference['native_report_sha256']):
        raise AssertionError('Frozen separately stored native reference audit changed')
    def records(path):return [DocumentScore(v['domain'],v['id'],v['group'],v['nll_sum'],v['real_targets']) for v in read(path)['records']]
    left=records(docs);right=records(directory/'document-scores.json')
    if abs(summarize(left,WEIGHTS)['weighted_nll']-reference['weighted_nll'])>1e-12:
        raise AssertionError('Reference domain sums do not reproduce the exact weighted score')
    return paired_group_interval(left,right,draws=10000,seed=71239,weights=WEIGHTS)


def run(root):
    out=root/IDENTITY;plan_file=out/'plan.json';plan=read(plan_file)
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal MAIN;explicit registered recovery required')
    if (file_sha256(root/TUNE/'report.json')!=plan['tune_report_sha256']
            or file_sha256(root/PREP/'manifest.json')!=plan['preparation_manifest_sha256']):
        raise AssertionError('Frozen selected TUNE/preparation changed')
    execution=MainExecution(root,plan_file,out/'execution');cpu=cf.ThreadPoolExecutor(2)
    started=time.time();host_cpu=time.process_time();lock=threading.Lock()
    states={};gates={};fits={};scores={};failures=[];result=dict(status='STOPPED_LATE_TWO_ALLOCATION_MAIN')
    def interrupted(number,frame):execution.failed.set();raise KeyboardInterrupt('Paired MAIN interrupted: '+str(number))
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    def observed(label,function,*args):
        try:return function(*args)
        except BaseException as error:
            execution.failed.set()
            with lock:
                failures.append(dict(phase=label,unix=time.time(),error=str(error),kind=type(error).__name__,traceback=traceback.format_exc()))
                write_state(out/'phase-failures.json',dict(failures=failures))
            raise
    def fit(key,phase,end,parent=None):
        name=IDENTITY+'-'+key+'-'+phase
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,
            '--case',key,'--segment-end',str(end),'--comm-port',str(plan['root_comm_ports'][name])]
        if parent:
            prior=checkpoint(parent);opts+=['--resume','/work/instance-nvme/runs/'+parent[0]+f'/checkpoint-u{prior["update"]}.pt','--resume-sha',prior['checkpoint_sha256']]
        row=execution.native(name,'scripts.train_future_span_primitives',opts,'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT')
        case=plan['cases'][key]
        if (row['case_id']!=key or row['plan_sha256']!=execution.sha or row['update']!=end or row['qualification_only']
                or row['objective_weight']!=0. or row['training_parameters']!=case['training_parameters']
                or row['deployed_parameters']!=141963840
                or row['actual_optimizer_calls_per_replica']!=end-(parent[1]['update'] if parent else 0)):
            raise AssertionError('Exact counted scientific segment identity/initialization differs')
        if parent and (row['recovered_state_digest']!=prior['full_state_digest'] or not row['complete_state_on_load_exact'] or not row['cpu_rng_on_load_exact']):
            raise AssertionError('Complete prefix/switch restoration differs')
        if end==4400 and ((row['real_targets'],row['valid_anchors'],row['estimated_dense_matrix_ops'])!=(case['end_targets'],case['end_anchors'],case['training_dense_matrix_ops']) or row['progress'][-1]['lr']!=0.):
            raise AssertionError('Final complete cooldown/source/work endpoint differs')
        capture=(name,row);state=verify_readback(root,plan_file,key,capture)
        with lock:
            states.setdefault(key,{})[str(end)]=state
            write_state(PUBLIC/'state-readbacks.json',states)
        return capture
    def gradient(key,capture,phase):
        name,row=capture;point=checkpoint(capture);identity=IDENTITY+'-'+key+'-'+phase
        common=['--plan-sha',execution.sha,'--case',key,'--checkpoint-sha',point['checkpoint_sha256']]
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',*common,
            '--checkpoint','/work/instance-nvme/runs/'+name+f'/checkpoint-u{row["update"]}.pt','--comm-port',str(plan['root_comm_ports'][identity])]
        native=execution.native(identity,'scripts.capture_future_span_primitives',opts,'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT')
        target=out/'gradients'/(key+'-'+phase)
        opts=['--root',str(root),'--plan',str(plan_file),*common,'--checkpoint',str(root/'runs'/name/f'checkpoint-u{row["update"]}.pt'),
            '--capture',str(root/'runs'/identity),'--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        independent=execution.cpu(key+'-'+phase,'scripts.audit_future_span_gradient',opts,target/'report.json','PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT')
        gate=dict(native=native,independent_cpu=independent);gradient_gate(gate,key,row['update'],point['checkpoint_sha256'])
        with lock:
            gates.setdefault(key,{})[phase]=gate
            write_state(PUBLIC/'gradient-progress.json',dict(completed={k:list(v) for k,v in gates.items()}))
        return gate
    def score(key,capture):
        name,row=capture;point=checkpoint(capture);target=out/'scores'/key
        opts=['--root',str(root),'--plan',str(plan_file),'--plan-sha',execution.sha,'--case',key,'--checkpoint',str(root/'runs'/name/'checkpoint-u4400.pt'),
            '--checkpoint-sha',point['checkpoint_sha256'],'--cohort','curve','--depth','2','--output',str(target),
            '--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        row=execution.cpu(key+'-score','scripts.evaluate_fixed_two_curve_cpu',opts,target/'report.json','COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE')
        final_score_gate(row,plan,key,point['checkpoint_sha256'])
        return row
    def path(key):
        prefix=observed(key+'-prefix',fit,key,'prefix',121)
        observed(key+'-prefix-gradient',gradient,key,prefix,'prefix-gradient')
        switch=observed(key+'-one-finish',fit,key,'one-finish',3300,prefix)
        # This audit uses depth_table[3300]=[2,2] on the exact unchanged
        # pre-switch checkpoint,before ANY deep-training update3301.
        if states[key]['3300']['next_forward_backward_depth']!=[2,2]:raise AssertionError('First deep objective differs')
        observed(key+'-switch-gradient',gradient,key,switch,'switch-gradient')
        final=observed(key+'-two-finish',fit,key,'two-finish',4400,switch)
        with lock:fits[key]=final[1]
        score_job=cpu.submit(observed,key+'-score',score,key,final)
        score_job.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
        observed(key+'-final-gradient',gradient,key,final,'final-gradient')
        value=score_job.result()
        with lock:scores[key]=value
        return value
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_PAIRED_COUNTED_LATE_ALLOCATION_MAIN',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as pool:
            jobs={key:pool.submit(path,key) for key in plan['cases']}
            for future in jobs.values():future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
            for key,future in jobs.items():scores[key]=future.result()
        comparisons={};gains={}
        for key,case in plan['cases'].items():
            seed=case['seed'];directory=out/'scores'/key
            comparisons[str(seed)]={name:compare(root,reference,directory) for name,reference in plan['references'][str(seed)].items()}
            gains[seed]=comparisons[str(seed)]['ordinary8']['gain_nats']
            if abs(gains[seed]-(plan['references'][str(seed)]['ordinary8']['weighted_nll']-scores[key]['metrics']['weighted_nll']))>1e-12:
                raise AssertionError('Independent document totals and endpoint scores differ')
            if scores[key]['checkpoint_sha256']!=gates[key]['final-gradient']['independent_cpu']['checkpoint_sha256']:
                raise AssertionError('Scoring and final audit checkpoints differ')
        summary=seed_summary(gains)
        result=dict(status='COMPLETE_QUALIFIED_LATE_TWO_ALLOCATION_MAIN',scores=scores,gradient_gates=gates,
            state_readbacks=states,completed_fit_reports=fits,paired_comparisons=comparisons,primary_seed_summary=summary,
            scientific_optimizer_updates=2*4400,selected_rate=plan['selected_rate'],
            further_control_preparation_eligible=summary['development_point_gate'],
            conventional_growth_and_equal_source_controls_still_required=True,fresh_confirmation_admitted=False)
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),error_kind=type(error).__name__,traceback=traceback.format_exc(),
            phase_failures=failures,completed_fit_reports=fits,scores=scores,gradient_gates=gates,state_readbacks=states)
        write_state(out/'initiating-failure.json',result);raise
    finally:
        execution.failed.set();cpu.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=execution.sha,scope=plan['scope'],wall_seconds=time.time()-started,
            host_controller_cpu_seconds=time.process_time()-host_cpu,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if not (a.root/IDENTITY/'plan.json').exists():register(a.root)
    if not a.prepare_only:run(a.root)


if __name__=='__main__':main()
