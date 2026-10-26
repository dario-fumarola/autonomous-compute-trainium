"""Bounded predictive tuning of qualified lean mixing and ordinary8 controls."""
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

import numpy as np

from compute_trainium.execution_isolation import managed_running
from compute_trainium.fixed_two_curve import stop_on_failed_future
from compute_trainium.future_span_execution import Execution
from compute_trainium.lean_baseline_science import EXECUTION,ROLES,signature
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.parcae_scientific_state import schedule_identity
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_fixed_control_replication import reserves

IDENTITY='lean-baseline-tune-v1'
PREP='lean-baseline-control-preparation-v2'
PUBLIC=Path('results/lean_baseline_controls_20261026/tune')
DOC='docs/LEAN_BASELINE_PREDICTIVE_PLAN_20261026.md'
ROLES_TO_TUNE=['mixing','ordinary8']


def read(path):return json.loads(Path(path).read_text())


def project_seconds(qualified):
    rates={p:qualified['records'][p]['warm']['warm_seconds']/100 for p in ROLES_TO_TUNE}
    # Three paired full trials; six cold starts plus two counted prefix starts.
    parts=dict(training=3*576*max(rates.values())*1.5,cold_starts=4*300,
        independent_scores=2*360,winner_captures=300,winner_full_gradient_audits=900,margin=300)
    return dict(warm_seconds_per_update=rates,parts=parts,complete_seconds=sum(parts.values()))


def register(root):
    if managed_running():raise RuntimeError('All numerical qualification workers must be closed')
    reserves(root)
    prep_file=root/PREP/'manifest.json';prep=read(prep_file)
    qual_file=root/'baseline-numerical-recovery-v1/report.json';qualified=read(qual_file)
    if (prep['status']!='PASS_LEAN_BASELINE_CONTROL_CPU_PREPARATION_ONLY'
            or qualified['status']!='PASS_THREE_BASELINE_NUMERICAL_RECOVERY_PROFILES'
            or prep['qualification_report_sha256']!=file_sha256(qual_file)):
        raise ValueError('Exact completed initializer and three-profile numerical receipts required')
    estimate=project_seconds(qualified)
    if estimate['complete_seconds']>7200:raise RuntimeError('Complete measured tuning stages exceed two hours')
    # Eight training phases and two selected zero-update captures, not every
    # possible communication-port identity. Include complete source snapshots.
    inode=project_snapshot_inodes(Path.cwd(),10)
    if os.statvfs(root).f_favail<inode['required_free_inodes']:raise RuntimeError('Complete tuning inode projection fails reserve')
    storage=19*1024**3
    if shutil.disk_usage(root).free<35*1024**3+storage:raise RuntimeError('Complete tuning artifacts plus protected space required')
    out=root/IDENTITY
    if out.exists():raise FileExistsError('Preserve tuning identity; explicit recovery required')
    extras=['scripts/run_lean_baseline_tune.py','compute_trainium/lean_baseline_science.py',
        'scripts/train_lean_baseline.py','scripts/evaluate_lean_baseline.py',
        'scripts/capture_lean_baseline_gradient.py','scripts/audit_lean_baseline_gradient.py',
        'scripts/train_low_depth_scientific.py','scripts/evaluate_low_depth_cpu.py',
        'scripts/capture_low_depth_gradient.py','scripts/audit_low_depth_gradient.py',
        'compute_trainium/low_depth_gradient_probe.py','compute_trainium/parcae_scientific_state.py',
        'compute_trainium/parcae_scientific_schedule.py','tests/test_lean_baseline_science.py',
        'tests/test_lean_baseline_tune.py',DOC]
    sources={**prep['source_sha256'],**{f:file_sha256(f) for f in extras}}
    if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Original qualified or new exact source list changed')
    worker,host=split_snapshot_sources(sources)
    keys=[role+'-lr'+str(i) for i in range(3) for role in ROLES_TO_TUNE]
    names=[IDENTITY+'-'+k+'-'+phase for k in keys for phase in ['prefix','finish','gradient']]
    ports=reserve_ports(root,names)
    original=read(root/'fixed-two-ntp-main-v1/plan.json')
    routing=original['routing_seed'];out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False)
    parent_plans={}
    for role in ROLES_TO_TUNE:
        profile=prep['profiles'][role];cases={};initial_key='s4127-'+('ordinary8' if role=='ordinary8' else 'family')
        initial=copy.deepcopy(prep['initial_states'][initial_key]);initial['file']='../../'+PREP+'/'+initial['file']
        for i,lr in enumerate(prep['lr_grid']):
            key=role+'-lr'+str(i);checkpoints=[2,576] if i==0 else [576]
            cases[key]=dict(arm=profile['arm'],seed=4127,state_seed=6136,parameters=profile['parameters'],
                initial_state=initial_key,peak_lr=lr,warmup_targets=50000000,
                cooldown_start_update=288,cooldown_start_targets=profile['tuning_cooldown_start_targets'],
                total_updates=576,end_targets=profile['tuning_targets'],
                depth_table='../../'+PREP+'/'+profile['depths_file'],depth_table_sha256=profile['depths_sha256'],
                work_table='../../'+PREP+'/'+profile['work_file'],work_table_sha256=profile['work_sha256'],
                checkpoint_updates=checkpoints,segment_ends=checkpoints,gradient_updates=[576],
                training_matmuls=576*profile['total_update_work'],primary_inference_depth=profile['depth'])
        plan=dict(identity=IDENTITY+'-'+role,purpose='natural_low_depth_scientific_pilot',
            launch_gates_complete=True,model_variant=role,execution_variant=EXECUTION,
            registered_unix=time.time(),model=profile['model'],precision='fp16',accumulation=16,
            routing_seed=routing,data=prep['data'],evaluation=prep['evaluation'],
            initial_states={initial_key:initial},cases=cases,source_sha256=sources,
            worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=ports,
            seconds_limit=7200,phase_seconds_limit=1800,cpu_phase_seconds_limit=1200,
            preparation_sha256=file_sha256(prep_file),qualification_sha256=file_sha256(qual_file),
            supported_evaluation_depth=ROLES[role][1],no_auxiliary_head=True,
            evaluation_noise_policy='window_position_only_fixed1024_draw_then_prefix_slice',scope=Path(DOC).read_text())
        folder=out/role;folder.mkdir();write_state(folder/'plan.json',plan)
        public=PUBLIC/role;public.mkdir();write_state(public/'plan.json',plan)
        parent_plans[role]=dict(file=role+'/plan.json',sha256=file_sha256(folder/'plan.json'))
    plan=dict(identity=IDENTITY,purpose='bounded_stronger_baseline_tuning',registered_unix=time.time(),
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        parent_plans=parent_plans,root_comm_ports=ports,seconds_limit=7200,phase_seconds_limit=1800,
        cpu_phase_seconds_limit=1200,measured_budget=estimate,inode_projection=inode,
        projected_artifact_bytes=storage,tuning_seed=4127,lr_grid=prep['lr_grid'],
        case_order=keys,actual_planned_scientific_updates=3456,native_training_phases=8,selected_gradient_captures=2,
        test_accessed=False,terminal_alpha_spent=0,automatic_main_admission=False,
        selection='Lowest fixed weighted TUNE NLL per role; ties choose lower learning rate. All trials complete cooldown. Selected mature full512 gradients must pass original2% family gates.',
        scope=Path(DOC).read_text())
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan


class GuardedExecution(Execution):
    def check(self):
        super().check();reserves(self.root)
        for item in self.plan['parent_plans'].values():
            if file_sha256(self.plan_file.parent/item['file'])!=item['sha256']:raise AssertionError('Exact saved child science plan changed')


def verify_prefix(root,parent_file,key,capture):
    import torch
    from scripts.qualify_native import digest
    name,row=capture;plan=read(parent_file);case=plan['cases'][key]
    checkpoint=root/'runs'/name/'checkpoint-u2.pt';record=next(r for r in row['checkpoint_records'] if r['update']==2)
    if file_sha256(checkpoint)!=record['checkpoint_sha256']:raise AssertionError('Counted scientific prefix bytes changed')
    state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if (state['signature']!=signature(file_sha256(parent_file),plan,case) or state['schedule']!=schedule_identity(case)
            or state['update']!=2 or state['cursor']!=1024 or not state['stable_trunk']
            or state['qualification_only'] is not False or len(state['cpu_rng_by_rank'])!=32
            or state['case_id']!=key or row['actual_optimizer_calls_per_replica']!=2
            or row['initial_model_digest']!=plan['initial_states'][case['initial_state']]['model_digest']):
        raise AssertionError('Exact counted scientific prefix identity/state/RNG differs')
    counts=np.load(root/plan['data']['packed']/'prefix_counts.npy',mmap_mode='r')
    work=np.load(parent_file.parent/case['work_table'],mmap_mode='r')
    domains=list(map(int,counts[1024]-counts[0]));expected=int(work[:2].sum())
    if (state['domain_targets']!=domains or state['real_targets']!=sum(domains)
            or state['executed_matmuls']!=expected or row['executed_matmuls']!=expected
            or digest({'model':dict(sorted(state['model'].items())),'optimizer':state['optimizer']})!=state['model_optimizer_digest']
            or state['model_optimizer_digest']!=record['full_state_digest']):raise AssertionError('Independent prefix state/source/work readback failed')
    return dict(checkpoint_sha256=record['checkpoint_sha256'],full_state_digest=record['full_state_digest'],
        update=2,source_cursor=1024,domain_targets=domains,executed_matmuls=expected,
        complete_state_readback_exact=True,all32_rng_states_present=True,replicas_equal=True)


def run(root):
    out=root/IDENTITY;file=out/'plan.json';plan=read(file)
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal tuning result')
    execution=GuardedExecution(root,file,out/'execution');cpu=cf.ThreadPoolExecutor(3)
    result=dict(status='STOPPED_LEAN_BASELINE_TUNING');prefixes={};fits={};scores={};gates={};failures=[]
    failure_lock=threading.Lock();score_jobs={};plans={r:read(out/p['file']) for r,p in plan['parent_plans'].items()}
    def interrupted(number,frame):
        write_state(out/'signal.json',dict(signal=number,unix=time.time()));execution.failed.set();raise KeyboardInterrupt('Owned lean baseline tuning interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def observed(label,function,*args):
        try:return function(*args)
        except BaseException as error:
            with failure_lock:
                failures.append(dict(phase=label,unix=time.time(),kind=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
                write_state(out/'phase-failures.json',dict(failures=failures))
                write_state(PUBLIC/'failure-status.json',dict(status='STOPPING_ON_PHASE_FAILURE',first_failure=failures[0],assigned_chips_may_be_stopping=True))
            execution.failed.set();raise
    def parent(key):
        role=key.rsplit('-lr',1)[0];return role,out/plan['parent_plans'][role]['file'],plans[role]
    def checkpoint(capture,end):
        name,row=capture;point=next(r for r in row['checkpoint_records'] if r['update']==end)
        return root/'runs'/name/f'checkpoint-u{end}.pt',point
    def fit(key,phase,end,prefix=None):
        role,parent_file,p=parent(key);name=IDENTITY+'-'+key+'-'+phase;case=p['cases'][key];sha=file_sha256(parent_file)
        options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+str(parent_file.relative_to(root)),
            '--plan-sha',sha,'--case',key,'--segment-end',str(end),'--comm-port',str(plan['root_comm_ports'][name])]
        previous=None
        if prefix:
            path,previous=checkpoint(prefix,2)
            options+=['--resume','/work/instance-nvme/'+str(path.relative_to(root)),'--resume-sha',previous['checkpoint_sha256']]
        row=execution.native(name,'scripts.train_lean_baseline',options,'COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT')
        if (row['plan_sha256']!=sha or row['case_id']!=key or row['update']!=end or row['qualification_only']
                or row['parameters']!=case['parameters'] or row['actual_optimizer_calls_per_replica']!=end-(2 if prefix else 0)
                or any(not r['all_complete_replica_states_equal'] for r in row['checkpoint_records'])):raise AssertionError('Frozen fit/replica/update identity differs')
        if previous and (not row['complete_state_on_load_exact'] or row['recovered_state_digest']!=previous['full_state_digest']):
            raise AssertionError('Complete counted prefix state restoration failed')
        if end==576 and (row['real_targets']!=case['end_targets'] or row['executed_matmuls']!=case['training_matmuls']):
            raise AssertionError('Completed own-cooldown endpoint counters differ')
        return name,row
    def score(key,capture):
        role,parent_file,p=parent(key);path,point=checkpoint(capture,576);target=out/'scores'/key
        opts=['--root',str(root),'--plan',str(parent_file),'--plan-sha',file_sha256(parent_file),
            '--case',key,'--checkpoint',str(path),'--checkpoint-sha',point['checkpoint_sha256'],'--cohort','tune',
            '--depth',str(ROLES[role][1]),'--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        return execution.cpu(key+'-score','scripts.evaluate_lean_baseline',opts,target/'report.json','COMPLETE_INDEPENDENT_LOW_DEPTH_DEVELOPMENT_SCORE')
    def gradient(key,capture):
        role,parent_file,p=parent(key);path,point=checkpoint(capture,576);name=IDENTITY+'-'+key+'-gradient'
        folder=out/'probes'/key;folder.mkdir(parents=True,exist_ok=False)
        probe=dict(identity=name,purpose='zero_update_low_depth_gradient_validation',parent_plan=str(parent_file.relative_to(root)),
            parent_plan_sha256=file_sha256(parent_file),points={key:dict(case_id=key,checkpoint=str(path.relative_to(root)),checkpoint_sha256=point['checkpoint_sha256'])},
            source_sha256=plan['source_sha256'],worker_source_sha256=plan['worker_source_sha256'],root_comm_ports=plan['root_comm_ports'])
        probe_file=folder/'plan.json';write_state(probe_file,probe);sha=file_sha256(probe_file)
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+str(probe_file.relative_to(root)),
            '--plan-sha',sha,'--point',key,'--comm-port',str(plan['root_comm_ports'][name])]
        native=execution.native(name,'scripts.capture_lean_baseline_gradient',opts,'CAPTURED_ZERO_UPDATE_LOW_DEPTH_GLOBAL_GRADIENT')
        target=out/'gradients'/key
        opts=['--root',str(root),'--plan',str(probe_file),'--plan-sha',sha,'--point',key,
            '--capture',str(root/'runs'/name),'--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
        gold=execution.cpu(key+'-gradient','scripts.audit_lean_baseline_gradient',opts,target/'report.json','PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT')
        return dict(native={k:v for k,v in native.items() if k not in ['probe_rows','noise_sha256_by_rank']},independent_cpu=gold)
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_COUNTED_LEAN_BASELINE_PREFIXES',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as pool:
            jobs={pool.submit(observed,role+'-prefix',fit,role+'-lr0','prefix',2):role+'-lr0' for role in ROLES_TO_TUNE}
            for future in jobs:future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
            for future in cf.as_completed(jobs):
                key=jobs[future];prefixes[key]=future.result()
        checked={k:verify_prefix(root,parent(k)[1],k,v) for k,v in prefixes.items()}
        write_state(PUBLIC/'counted-prefixes.json',dict(status='PASS_COMPLETE_COUNTED_PREFIX_READBACK',records=checked,
            strict_future_trajectory_identity_established=False,scope='The two updates count inside each trial. Fresh-process continuation must restore exact model/optimizer/RNG state; bit-exact future trajectories are not asserted.'))
        remaining=plan['measured_budget']['complete_seconds']-600
        if remaining>execution.deadline-time.time():raise RuntimeError('Remaining complete frozen tuning stages no longer fit original cap')
        write_state(PUBLIC/'status.json',dict(status='RUNNING_LEAN_BASELINE_THREE_RATE_TUNING',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as pool:
            jobs={pool.submit(observed,key+'-fit',fit,key,'finish',576,prefixes.get(key)):key for key in plan['case_order']}
            for future in jobs:future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
            for future in cf.as_completed(jobs):
                key=jobs[future];fits[key]=future.result();score_jobs[key]=cpu.submit(observed,key+'-score',score,key,fits[key])
                score_jobs[key].add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
                write_state(PUBLIC/'fit-progress.json',dict(completed_fit_reports={k:v[1] for k,v in fits.items()}))
        for key,future in score_jobs.items():scores[key]=future.result()
        selected={r:min((k for k in scores if k.startswith(r+'-lr')),key=lambda k:(scores[k]['metrics']['weighted_nll'],plans[r]['cases'][k]['peak_lr'])) for r in ROLES_TO_TUNE}
        write_state(PUBLIC/'selected-pending-gates.json',dict(selected=selected,scores=scores,numerical_admission_complete=False))
        with cf.ThreadPoolExecutor(2) as pool:
            jobs={pool.submit(observed,key+'-mature-gradient',gradient,key,fits[key]):role for role,key in selected.items()}
            for future in jobs:future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
            for future in cf.as_completed(jobs):gates[jobs[future]]=future.result()
        actual=sum(r[1]['actual_optimizer_calls_per_replica'] for r in [*prefixes.values(),*fits.values()])
        if actual!=3456:raise AssertionError('Counted tuning optimizer updates differ')
        execution.check();result=dict(status='COMPLETE_QUALIFIED_LEAN_BASELINE_TUNING',selected=selected,
            selected_rates={r:plans[r]['cases'][k]['peak_lr'] for r,k in selected.items()},
            boundary_winners={r:plans[r]['cases'][k]['peak_lr'] in [.008,.032] for r,k in selected.items()},
            scores=scores,selected_final_gates=gates,counted_prefixes=checked,
            scientific_optimizer_updates=actual,completed_fit_reports={k:v[1] for k,v in fits.items()},automatic_main_admission=False)
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),error_kind=type(error).__name__,phase_failures=failures,
            prefix_reports={k:v[1] for k,v in prefixes.items()},completed_fit_reports={k:v[1] for k,v in fits.items()},scores=scores,selected_final_gates=gates)
        raise
    finally:
        execution.failed.set();cpu.shutdown(wait=True,cancel_futures=True);execution.close(PUBLIC/'execution-costs.json')
        for s,handler in handlers.items():signal.signal(s,handler)
        result.update(plan_sha256=execution.sha,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if not (a.root/IDENTITY/'plan.json').exists():register(a.root)
    if not a.prepare_only:run(a.root)


if __name__=='__main__':main()
