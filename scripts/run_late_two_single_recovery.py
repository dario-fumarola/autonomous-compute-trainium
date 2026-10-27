"""Bounded completion of the intact original seed; no paired advancement."""
import argparse
import concurrent.futures as cf
import copy
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
from compute_trainium.future_span_scientific_state import signature
from compute_trainium.late_allocation_main import gradient_gate, final_score_gate
from compute_trainium.late_allocation_recovery import rebind_same_endpoint, original_deadline
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint, file_sha256
from scripts.qualify_native import digest
from scripts.run_fixed_control_replication import reserves
from scripts.run_fixed_two_ntp_tune import GuardedExecution
from scripts.run_late_two_main import read, checkpoint, verify_readback, compare

IDENTITY='late-two-single-recovery-v1'
PARENT='late-two-allocation-main-v1'
TUNE='late-two-allocation-tune-v1'
PREP='late-two-allocation-preparation-v1'
KEY='late-two-ntp-s5231'
PUBLIC=Path('results/late_two_allocation_20261026/single-recovery')
DOC='docs/LATE_TWO_SINGLE_RECOVERY_20261027.md'


def register(root):
    import torch
    now=time.time(); torch.set_num_threads(12)
    if managed_running():raise RuntimeError('All original owners must be stopped')
    old_file=root/PARENT/'plan.json'; old=read(old_file); old_sha=file_sha256(old_file)
    parent=read(root/PARENT/'report.json'); clock=read(root/PARENT/'execution/execution-start.json')
    if (parent['status']!='STOPPED_LATE_TWO_ALLOCATION_MAIN' or parent['plan_sha256']!=old_sha
            or clock['plan_sha256']!=old_sha):
        raise AssertionError('Exact stopped original MAIN required')
    if any(file_sha256(p)!=sha for p,sha in old['source_sha256'].items()):
        raise AssertionError('Original qualified source changed')
    phase=PARENT+'-'+KEY+'-one-finish'; row=read(root/'runs'/phase/'report.json')
    capture=(phase,row); state=verify_readback(root,old_file,KEY,capture)
    if row['update']!=3300 or row['status']!='COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT':
        raise AssertionError('Exact completed switch checkpoint required')
    # Recovery is selected by state validity before observing any endpoint quality.
    failed=root/'runs'/(PARENT+'-late-two-ntp-s17293-one-finish')
    replicas=[read(p) for p in failed.glob('replica-u3300-rank*.json')]
    if len(replicas)!=32 or len({v['full_state_digest'] for v in replicas})!=2 or (failed/'checkpoint-u3300.pt').exists():
        raise AssertionError('Preserved failed replica evidence required')
    tune=read(root/TUNE/'report.json')
    two=max(v['two_pass_seconds']/100 for v in tune['actual_warm'].values())
    audit_file=root/'fixed-two-ntp-main-v1/report.json'
    audit_record=read(audit_file)['independent_gradient_gates']['fixed-2b']['independent_cpu']
    if audit_record['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT':
        raise AssertionError('Measured complete mature audit required')
    audit=audit_record['wall_seconds']
    parts=dict(two_training=1100*two*1.5,three_native_cold=3*250,
        two_full_cpu_audits=2*audit*1.5,independent_score=600,closure=300)
    estimate=sum(parts.values())
    deadline=original_deadline(now,clock['deadline_unix'],old['original_composite_deadline_unix'])
    if now+estimate+120>deadline:raise RuntimeError('Complete single recovery does not fit original clocks')
    reserves(root,admission=True)
    if shutil.disk_usage(root).free<75*1024**3:raise RuntimeError('Complete artifacts plus reserve required')
    inodes=project_snapshot_inodes(Path.cwd(),3)
    if os.statvfs(root).f_favail<inodes['required_free_inodes']:raise RuntimeError('All remaining snapshot inodes required')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve partial/terminal recovery')
    extras=['scripts/run_late_two_single_recovery.py','compute_trainium/late_allocation_recovery.py',
        'tests/test_late_allocation_recovery.py',DOC]
    sources={**old['source_sha256'],**{p:file_sha256(p) for p in extras}}
    worker,host=split_snapshot_sources(sources)
    plan=copy.deepcopy(old)
    plan.update(identity=IDENTITY,registered_unix=now,cases={KEY:copy.deepcopy(old['cases'][KEY])},
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,[IDENTITY+'-'+KEY+'-'+p for p in ['switch-gradient','two-finish','final-gradient']]),
        seconds_limit=7200,phase_seconds_limit=3000,cpu_phase_seconds_limit=1200,
        original_main_deadline_unix=clock['deadline_unix'],original_parent_plan_sha256=old_sha,
        original_parent_report_sha256=file_sha256(root/PARENT/'report.json'),
        measured_budget=dict(parts=parts,complete_seconds=estimate,two_seconds_per_update=two,
            mature_cpu_audit_seconds=audit,mature_audit_source_sha256=file_sha256(audit_file)),complete_source_inode_projection=inodes,
        stage='descriptive-single-recovery',primary_scope='One intact reused development seed; paired gate cannot be evaluated',
        scope=Path(DOC).read_text(),further_control_preparation_eligible=False)
    out.mkdir();PUBLIC.mkdir(parents=True); file=out/'plan.json';write_state(file,plan)
    point=checkpoint(capture); original=root/'runs'/phase/'checkpoint-u3300.pt'
    saved=torch.load(original,map_location='cpu',mmap=True,weights_only=False)
    new_sha=file_sha256(file)
    rebound=rebind_same_endpoint(saved,signature(old_sha,old,old['cases'][KEY]),
        signature(new_sha,plan,plan['cases'][KEY]),old['cases'][KEY],plan['cases'][KEY],case_id=KEY,update=3300)
    before=digest({k:v for k,v in saved.items() if k!='signature'})
    anchor_name=IDENTITY+'-signature-only-anchor'; anchor=root/'runs'/anchor_name;anchor.mkdir()
    migrated=anchor/'checkpoint-u3300.pt';atomic_checkpoint(migrated,rebound)
    check=torch.load(migrated,map_location='cpu',mmap=True,weights_only=False)
    if digest({k:v for k,v in check.items() if k!='signature'})!=before or check['signature']!=rebound['signature']:
        raise AssertionError('Signature migration changed original state')
    derived=copy.deepcopy(point);derived['checkpoint_sha256']=file_sha256(migrated)
    derived_capture=dict(update=3300,checkpoint_records=[derived])
    migration=dict(status='PASS_SIGNATURE_ONLY_SAME_ENDPOINT_RECOVERY',
        original_checkpoint_sha256=point['checkpoint_sha256'],new_checkpoint_sha256=derived['checkpoint_sha256'],
        original_plan_sha256=old_sha,new_plan_sha256=new_sha,all_other_state_exact=True,
        inherited_replica_agreement=True,anchor_directory=anchor_name,derived_capture=derived_capture,
        scope='Derived checkpoint metadata; not a Native phase report. Original model/optimizer/RNG/counters/schedule unchanged.')
    verify_readback(root,file,KEY,(anchor_name,derived_capture))
    write_state(out/'migration.json',migration);write_state(PUBLIC/'migration.json',migration)
    write_state(PUBLIC/'plan.json',plan)
    write_state(PUBLIC/'status.json',dict(status='PREPARED_SINGLE_RECOVERY_NOT_ADMITTED',assigned_chips=0))
    return plan


def admit(root,file,plan):
    if managed_running():raise RuntimeError('Original owners must be terminal')
    if file_sha256(root/PARENT/'report.json')!=plan['original_parent_report_sha256']:
        raise AssertionError('Preserved predecessor changed')
    cutoff=original_deadline(time.time(),plan['original_main_deadline_unix'],plan['original_composite_deadline_unix'])
    if time.time()+plan['measured_budget']['complete_seconds']+120>cutoff:
        raise RuntimeError('Complete measured recovery no longer fits original clock')
    reserves(root,admission=True)
    migration=read(file.parent/'migration.json')
    verify_readback(root,file,KEY,(migration['anchor_directory'],migration['derived_capture']))


class RecoveryExecution(GuardedExecution):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.deadline=original_deadline(self.execution['started_unix'],
            self.plan['original_main_deadline_unix'],self.plan['original_composite_deadline_unix'])
        self.execution['deadline_unix']=self.deadline
        self.execution['original_main_deadline_unix']=self.plan['original_main_deadline_unix']
        write_state(self.out/'execution-start.json',self.execution)


def run(root):
    out=root/IDENTITY;plan_file=out/'plan.json';plan=read(plan_file)
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal MAIN;explicit registered recovery required')
    if (file_sha256(root/TUNE/'report.json')!=plan['tune_report_sha256']
            or file_sha256(root/PREP/'manifest.json')!=plan['preparation_manifest_sha256']):
        raise AssertionError('Frozen selected TUNE/preparation changed')
    admit(root,plan_file,plan)
    execution=RecoveryExecution(root,plan_file,out/'execution');cpu=cf.ThreadPoolExecutor(2)
    started=time.time();host_cpu=time.process_time();lock=threading.Lock()
    states={};gates={};fits={};scores={};failures=[];result=dict(status='STOPPED_LATE_TWO_SINGLE_RECOVERY')
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
        migration=read(out/'migration.json')
        switch=(migration['anchor_directory'],migration['derived_capture'])
        states[key]={'3300':verify_readback(root,plan_file,key,switch)}
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
        write_state(PUBLIC/'status.json',dict(status='RUNNING_DESCRIPTIVE_SINGLE_SEED_RECOVERY',maximum_chips=8))
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
        summary=dict(scope='Single intact seed only; original paired gate unevaluable',
            descriptive_gain_nats=gains[5231],development_point_gate=False,terminal_success=False)
        result=dict(status='COMPLETE_AUDITED_DESCRIPTIVE_LATE_TWO_SINGLE_RECOVERY',scores=scores,gradient_gates=gates,
            state_readbacks=states,completed_fit_reports=fits,paired_comparisons=comparisons,primary_seed_summary=summary,
            actual_new_optimizer_updates=1100,retained_single_trajectory_updates=4400,
            original_actual_updates_including_failed=6600,original_failed_discarded_updates=3179,
            total_campaign_actual_updates_per_replica=7700,original_paired_comparison_complete=False,selected_rate=plan['selected_rate'],
            further_control_preparation_eligible=False,
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
