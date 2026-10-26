"""One explicitly admitted paired seed of the frozen stronger-control study."""
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
from compute_trainium.lean_baseline_science import EXECUTION,ROLES
from compute_trainium.metrics import DocumentScore,paired_group_interval
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_fixed_control_replication import reserves
from scripts.run_lean_baseline_tune import GuardedExecution,verify_prefix,PREP

DOC='docs/LEAN_BASELINE_MAIN_20261026.md'
TUNE='lean-baseline-tune-v1'
ALL_ROLES=['mixing','ordinary8','one']


def read(path):return json.loads(Path(path).read_text())


def identity(seed):
    if seed not in [5231,17293]:raise ValueError('Only the two original paired development seeds are registered')
    return 'lean-baseline-main-s'+str(seed)+'-v1'


def selected_rates(tune,plans):
    if tune['status']!='COMPLETE_QUALIFIED_LEAN_BASELINE_TUNING':raise ValueError('Complete numerical-gated tuning required')
    rates={'one':.032}
    for role in ['mixing','ordinary8']:
        cases=plans[role]['cases']
        if set(cases)!={k for k in tune['scores'] if k.startswith(role+'-lr')}:raise AssertionError('Complete common three-rate opportunity required')
        winner=min(cases,key=lambda k:(tune['scores'][k]['metrics']['weighted_nll'],cases[k]['peak_lr']))
        gate=tune['selected_final_gates'][role]['independent_cpu']
        if (tune['selected'][role]!=winner or tune['selected_rates'][role]!=cases[winner]['peak_lr']
                or gate['status']!='PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT' or gate['case_id']!=winner):
            raise AssertionError('Qualified complete-grid selection differs')
        rates[role]=cases[winner]['peak_lr']
    return rates


def project_main_seconds(profiles,qualified):
    training=sum(profiles[r]['matched_work']['updates']*qualified['records'][r]['warm']['warm_seconds']/100 for r in ALL_ROLES)
    parts=dict(training=training*1.5/2,cold_training=3*300/2+300,
        captures=3*300/2,independent_scores=600,independent_full_gradients=1800,margin=600)
    return dict(parts=parts,complete_seconds=sum(parts.values()))


def frozen_reference(root,seed,evaluation):
    directory='fixed-two-ntp-main-v1' if seed==5231 else 'fixed-control-replication-main-v1'
    key='fixed-2b' if seed==5231 else 'fixed-s17293-data'
    file=root/directory/'scores'/key/'report.json';row=read(file);parent=root/directory/'report.json'
    field='independent_gradient_gates' if seed==5231 else 'gradient_gates'
    gate=read(parent)[field][key]['independent_cpu'];documents=file.parent/'document-scores.json'
    if (row['status']!='COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE'
            or row['seed']!=seed or row['depth']!=2 or row['cohort']!='curve'
            or row['training_update']!=3840 or row['training_real_targets']!=2009762680
            or row['source_corpus_sha256']!=evaluation['corpus_sha256']
            or gate['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
            or gate['checkpoint_sha256']!=row['checkpoint_sha256']):raise AssertionError('Exact audited paired fixed-two reference required')
    return dict(file=str(file.relative_to(root)),sha256=file_sha256(file),parent_report_sha256=file_sha256(parent),
        document_scores=str(documents.relative_to(root)),document_scores_sha256=file_sha256(documents),
        checkpoint_sha256=row['checkpoint_sha256'],score=row['metrics']['weighted_nll'])


def register(root,seed):
    name=identity(seed);out=root/name;public=Path('results/lean_baseline_controls_20261026/main')/str(seed)
    if out.exists():raise FileExistsError('Preserve each admitted main seed; recovery requires a new explicit identity')
    reserves(root)
    if managed_running():raise RuntimeError('All tuning/previous native owners must be closed')
    tune_file=root/TUNE/'report.json';tune=read(tune_file);tune_plan=read(root/TUNE/'plan.json')
    if tune['plan_sha256']!=file_sha256(root/TUNE/'plan.json'):raise AssertionError('Tuning receipt identity changed')
    tuning_plans={r:read(root/TUNE/v['file']) for r,v in tune_plan['parent_plans'].items()}
    rates=selected_rates(tune,tuning_plans)
    prep_file=root/PREP/'manifest.json';prep=read(prep_file)
    reference=frozen_reference(root,seed,prep['evaluation'])
    qualified_file=root/'baseline-numerical-recovery-v1/report.json';qualified=read(qualified_file)
    if (qualified['status']!='PASS_THREE_BASELINE_NUMERICAL_RECOVERY_PROFILES'
            or prep['qualification_report_sha256']!=file_sha256(qualified_file)
            or prep['status']!='PASS_LEAN_BASELINE_CONTROL_CPU_PREPARATION_ONLY'):
        raise AssertionError('Exact random initializers and numerical receipts required')
    estimate=project_main_seconds(prep['profiles'],qualified)
    if estimate['complete_seconds']>18000:raise RuntimeError('Complete seed including audits exceeds five-hour cap')
    inode=project_snapshot_inodes(Path.cwd(),7)
    if os.statvfs(root).f_favail<inode['required_free_inodes']:raise RuntimeError('All main snapshots/artifacts plus protected inode reserve required')
    if shutil.disk_usage(root).free<47*1024**3:raise RuntimeError('Twelve GiB complete main artifact allowance plus protected space required')
    sources={**tune_plan['source_sha256']}
    for f in ['scripts/run_lean_baseline_main.py','tests/test_lean_baseline_main.py',DOC]:sources[f]=file_sha256(f)
    if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Exact qualified stack or main sources changed')
    worker,host=split_snapshot_sources(sources)
    names=[name+'-'+r+'-'+phase for r in ALL_ROLES for phase in ['prefix','finish','gradient']]
    ports=reserve_ports(root,names);out.mkdir();public.mkdir(parents=True,exist_ok=False)
    parent_plans={};original=read(root/'fixed-two-ntp-main-v1/plan.json')
    for role in ALL_ROLES:
        profile=prep['profiles'][role];endpoint=profile['matched_work'];key=role+'-s'+str(seed)
        folder=out/role;folder.mkdir()
        initial_key='s'+str(seed)+'-'+('ordinary8' if role=='ordinary8' else 'family')
        initial=copy.deepcopy(prep['initial_states'][initial_key]);initial['file']='../../'+PREP+'/'+initial['file']
        if file_sha256(out/role/initial['file'])!=initial['sha256']:raise AssertionError('Original random scientific initializer changed')
        end=endpoint['updates'];checkpoints=[2,end] if role=='one' else [end]
        case=dict(arm=profile['arm'],seed=seed,state_seed=seed+2009,parameters=profile['parameters'],
            initial_state=initial_key,peak_lr=rates[role],warmup_targets=50000000,
            cooldown_start_update=endpoint['cooldown_start_update'],cooldown_start_targets=endpoint['cooldown_start_targets'],
            total_updates=end,end_targets=endpoint['real_targets'],depth_table='../../'+PREP+'/'+profile['depths_file'],
            depth_table_sha256=profile['depths_sha256'],work_table='../../'+PREP+'/'+profile['work_file'],
            work_table_sha256=profile['work_sha256'],checkpoint_updates=checkpoints,segment_ends=checkpoints,
            gradient_updates=[end],training_matmuls=endpoint['executed_matmuls'],primary_inference_depth=profile['depth'])
        child=dict(identity=name+'-'+role,purpose='natural_low_depth_scientific_pilot',launch_gates_complete=True,
            model_variant=role,execution_variant=EXECUTION,registered_unix=time.time(),model=profile['model'],
            precision='fp16',accumulation=16,routing_seed=original['routing_seed'],data=prep['data'],evaluation=prep['evaluation'],
            initial_states={initial_key:initial},cases={key:case},source_sha256=sources,worker_source_sha256=worker,
            host_only_source_sha256=host,root_comm_ports=ports,seconds_limit=18000,phase_seconds_limit=10800,
            cpu_phase_seconds_limit=1800,preparation_sha256=file_sha256(prep_file),tuning_report_sha256=file_sha256(tune_file),
            qualification_sha256=file_sha256(qualified_file),no_auxiliary_head=True,
            evaluation_noise_policy='window_position_only_fixed1024_draw_then_prefix_slice',scope=Path(DOC).read_text())
        write_state(folder/'plan.json',child)
        dst=public/role;dst.mkdir();write_state(dst/'plan.json',child)
        parent_plans[role]=dict(file=role+'/plan.json',sha256=file_sha256(folder/'plan.json'),case_id=key)
    plan=dict(identity=name,purpose='stronger_control_paired_development_seed',seed=seed,registered_unix=time.time(),
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,parent_plans=parent_plans,
        root_comm_ports=ports,seconds_limit=18000,phase_seconds_limit=10800,cpu_phase_seconds_limit=1800,
        measured_budget=estimate,inode_projection=inode,projected_artifact_bytes=12*1024**3,
        fixed_two_reference=reference,selected_rates=rates,tuning_report_sha256=file_sha256(tune_file),
        target_training_matrix_budget=prep['target_training_matrix_budget'],scope=Path(DOC).read_text(),
        test_accessed=False,terminal_alpha_spent=0,automatic_4b_admission=False)
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    return plan


def paired_comparison(root,reference,score_dir):
    left=root/reference['document_scores'];right=score_dir/'document-scores.json'
    if file_sha256(left)!=reference['document_scores_sha256']:raise AssertionError('Frozen recurrence pairing changed')
    def rows(file):
        return [DocumentScore(r['domain'],r['id'],r['group'],r['nll_sum'],r['real_targets']) for r in read(file)['records']]
    # Positive means the cheaper baseline has higher loss, hence recurrence wins.
    return paired_group_interval(rows(right),rows(left),seed=314159,draws=10000)


def run(root,seed):
    name=identity(seed);out=root/name;file=out/'plan.json';plan=read(file)
    public=Path('results/lean_baseline_controls_20261026/main')/str(seed)
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal main result')
    execution=GuardedExecution(root,file,out/'execution');cpu=cf.ThreadPoolExecutor(3);audits=cf.ThreadPoolExecutor(2)
    children={r:read(out/v['file']) for r,v in plan['parent_plans'].items()};fits={};scores={};gates={};comparisons={};failures=[]
    lock=threading.Lock();score_jobs={};gate_jobs={};prefix=None;result=dict(status='STOPPED_LEAN_BASELINE_MAIN_SEED')
    def interrupted(number,frame):
        write_state(out/'signal.json',dict(signal=number,unix=time.time()));execution.failed.set();raise KeyboardInterrupt('Owned main seed interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def observed(label,function,*args):
        try:return function(*args)
        except BaseException as error:
            with lock:
                failures.append(dict(phase=label,unix=time.time(),kind=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
                write_state(out/'phase-failures.json',dict(failures=failures))
                write_state(public/'failure-status.json',dict(status='STOPPING_ON_PHASE_FAILURE',first_failure=failures[0]))
            execution.failed.set();raise
    def case(role):
        entry=plan['parent_plans'][role];key=entry['case_id'];return out/entry['file'],key,children[role]['cases'][key]
    def point(capture,end):
        run_name,row=capture;r=next(v for v in row['checkpoint_records'] if v['update']==end)
        return root/'runs'/run_name/f'checkpoint-u{end}.pt',r
    def fit(role,phase,end,parent=None):
        parent_file,key,c=case(role);run_name=name+'-'+role+'-'+phase
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+str(parent_file.relative_to(root)),
            '--plan-sha',file_sha256(parent_file),'--case',key,'--segment-end',str(end),'--comm-port',str(plan['root_comm_ports'][run_name])]
        previous=None
        if parent:
            path,previous=point(parent,2);opts+=['--resume','/work/instance-nvme/'+str(path.relative_to(root)),'--resume-sha',previous['checkpoint_sha256']]
        row=execution.native(run_name,'scripts.train_lean_baseline',opts,'COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT')
        if (row['plan_sha256']!=file_sha256(parent_file) or row['case_id']!=key or row['qualification_only']
                or row['update']!=end or row['actual_optimizer_calls_per_replica']!=end-(2 if parent else 0)
                or any(not r['all_complete_replica_states_equal'] for r in row['checkpoint_records'])):
            raise AssertionError('Main exact counted endpoint/replica identity differs')
        if previous and (not row['complete_state_on_load_exact'] or row['recovered_state_digest']!=previous['full_state_digest']):
            raise AssertionError('Exact counted main prefix restoration failed')
        if end==c['total_updates'] and (row['real_targets']!=c['end_targets'] or row['executed_matmuls']!=c['training_matmuls']):
            raise AssertionError('Main source/work/complete-cooldown endpoint differs')
        return run_name,row
    def score(role,capture):
        parent_file,key,c=case(role);path,r=point(capture,c['total_updates']);target=out/'scores'/role
        opts=['--root',str(root),'--plan',str(parent_file),'--plan-sha',file_sha256(parent_file),'--case',key,
            '--checkpoint',str(path),'--checkpoint-sha',r['checkpoint_sha256'],'--cohort','curve','--depth',str(ROLES[role][1]),
            '--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1800']
        return execution.cpu(role+'-score','scripts.evaluate_lean_baseline',opts,target/'report.json','COMPLETE_INDEPENDENT_LOW_DEPTH_DEVELOPMENT_SCORE')
    def gradient(role,capture):
        parent_file,key,c=case(role);path,r=point(capture,c['total_updates']);run_name=name+'-'+role+'-gradient'
        folder=out/'probes'/role;folder.mkdir(parents=True,exist_ok=False)
        probe=dict(identity=run_name,purpose='zero_update_low_depth_gradient_validation',parent_plan=str(parent_file.relative_to(root)),
            parent_plan_sha256=file_sha256(parent_file),points={role:dict(case_id=key,checkpoint=str(path.relative_to(root)),checkpoint_sha256=r['checkpoint_sha256'])},
            source_sha256=plan['source_sha256'],worker_source_sha256=plan['worker_source_sha256'],root_comm_ports=plan['root_comm_ports'])
        probe_file=folder/'plan.json';write_state(probe_file,probe);sha=file_sha256(probe_file)
        opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+str(probe_file.relative_to(root)),
            '--plan-sha',sha,'--point',role,'--comm-port',str(plan['root_comm_ports'][run_name])]
        native=execution.native(run_name,'scripts.capture_lean_baseline_gradient',opts,'CAPTURED_ZERO_UPDATE_LOW_DEPTH_GLOBAL_GRADIENT')
        target=out/'gradients'/role
        opts=['--root',str(root),'--plan',str(probe_file),'--plan-sha',sha,'--point',role,'--capture',str(root/'runs'/run_name),
            '--output',str(target),'--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','1800']
        gold=execution.cpu(role+'-gradient','scripts.audit_lean_baseline_gradient',opts,target/'report.json','PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT')
        return dict(native={k:v for k,v in native.items() if k not in ['probe_rows','noise_sha256_by_rank']},independent_cpu=gold)
    try:
        write_state(public/'status.json',dict(status='RUNNING_COUNTED_LEAN_ONE_MAIN_PREFIX',maximum_chips=8))
        prefix=observed('one-counted-prefix',fit,'one','prefix',2)
        checked=verify_prefix(root,case('one')[0],case('one')[1],prefix)
        write_state(public/'counted-prefix.json',checked)
        write_state(public/'status.json',dict(status='RUNNING_WORK_MATCHED_STRONGER_CONTROLS',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as pool:
            jobs={pool.submit(observed,role+'-fit',fit,role,'finish',case(role)[2]['total_updates'],prefix if role=='one' else None):role for role in ALL_ROLES}
            for future in jobs:future.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
            for future in cf.as_completed(jobs):
                role=jobs[future];fits[role]=future.result()
                score_jobs[role]=cpu.submit(observed,role+'-score',score,role,fits[role])
                gate_jobs[role]=audits.submit(observed,role+'-mature-gradient',gradient,role,fits[role])
                for f in [score_jobs[role],gate_jobs[role]]:f.add_done_callback(lambda f:stop_on_failed_future(f,execution.failed))
                write_state(public/'fit-progress.json',dict(completed_fit_reports={r:v[1] for r,v in fits.items()}))
        for role,future in score_jobs.items():scores[role]=future.result()
        for role,future in gate_jobs.items():gates[role]=future.result()
        for role in ALL_ROLES:comparisons[role]=paired_comparison(root,plan['fixed_two_reference'],out/'scores'/role)
        actual=prefix[1]['actual_optimizer_calls_per_replica']+sum(v[1]['actual_optimizer_calls_per_replica'] for v in fits.values())
        if actual!=sum(case(role)[2]['total_updates'] for role in ALL_ROLES):raise AssertionError('Every counted main update must reconcile')
        execution.check();result=dict(status='COMPLETE_QUALIFIED_LEAN_BASELINE_MAIN_SEED',seed=seed,scores=scores,
            final_gradient_gates=gates,recurrence_advantage=comparisons,fixed_two_reference=plan['fixed_two_reference'],
            counted_prefix=checked,scientific_optimizer_updates=actual,completed_fit_reports={r:v[1] for r,v in fits.items()},
            all_point_estimates_exceed_useful_threshold=all(plan['fixed_two_reference']['score']+0.01<scores[r]['metrics']['weighted_nll'] for r in ALL_ROLES),
            automatic_4b_admission=False)
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),error_kind=type(error).__name__,phase_failures=failures,
            prefix_report=prefix[1] if prefix else None,completed_fit_reports={r:v[1] for r,v in fits.items()},scores=scores,final_gradient_gates=gates)
        raise
    finally:
        execution.failed.set();audits.shutdown(wait=True,cancel_futures=True);cpu.shutdown(wait=True,cancel_futures=True)
        execution.close(public/'execution-costs.json')
        for s,handler in handlers.items():signal.signal(s,handler)
        result.update(plan_sha256=execution.sha,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(public/'report.json',result);write_state(public/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();name=identity(a.seed)
    if not (a.root/name/'plan.json').exists():register(a.root,a.seed)
    if not a.prepare_only:run(a.root,a.seed)


if __name__=='__main__':main()
