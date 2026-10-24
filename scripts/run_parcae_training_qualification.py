"""Complete bounded training/profile/recovery checks after prior numerical gates."""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import torch

from compute_trainium.execution_isolation import cancel_worker_container,managed_running
from compute_trainium.owned_native_phase import create_worker
from compute_trainium.parcae_recovery_check import check_replay
from compute_trainium.parcae_training_candidate import gradient_comparison
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def launch_policy(gradient,optimizer,fp16,managed_active,now,latest):
    if now>latest:raise TimeoutError('Training qualification latest-start window expired')
    for record,status in [(gradient,'PASS_PAIRED_NATIVE_PARCAE_STATIC_STAGE_GLOBAL_GRADIENTS'),
                          (optimizer,'PASS_PAIRED_NATIVE_PARCAE_OPTIMIZER_AND_RECOVERY')]:
        if record and record['status']!=status:
            raise RuntimeError('Prior FP32/optimizer qualification failed; no training benchmark starts')
    if not gradient or not optimizer or not fp16 or managed_active:return None
    if fp16['status']=='PASS_PAIRED_NATIVE_PARCAE_FP16_BODY_GRADIENTS':return ['fp32','fp16']
    if fp16['status']=='STOPPED_NATIVE_PARCAE_FP16_BODY_QUALIFICATION':return ['fp32']
    raise RuntimeError('Unknown terminal precision status')


def finite_state(value):
    if isinstance(value,torch.Tensor):
        if not torch.isfinite(value).all():raise ValueError('Nonfinite model/optimizer state')
    elif isinstance(value,dict):
        for item in value.values():finite_state(item)
    elif isinstance(value,(list,tuple)):
        for item in value:finite_state(item)


def check_native_pair(primary,replay):
    a=json.loads((primary/'report.json').read_text());b=json.loads((replay/'report.json').read_text())
    if a['status']!='CAPTURED_PARCAE_TRAINING_QUALIFICATION' or b['status']!=a['status']:
        raise AssertionError('Training phase did not finish')
    if a['mode']!='warm' or b['mode']!='replay' or a['warm_updates']!=100:
        raise AssertionError('Warm100/replay contract differs')
    if not b['complete_state_on_load_exact'] or not b['per_rank_cpu_rng_on_load_exact'] or b['recovered_state_digest']!=a['checkpoint_state_digest']:
        raise AssertionError('Complete state/RNG recovery differs')
    if a['actual_optimizer_calls_per_replica']!=121 or b['actual_optimizer_calls_per_replica']!=1:
        raise AssertionError('Actual update count differs')
    for key in ['signature','cursor','total_real_targets','model_matmuls','optimizer_matmuls','initial_model_digest']:
        if a[key]!=b[key]:raise AssertionError('Source/work/signature differs after replay: '+key)
    for report,path in [(a,primary),(b,replay)]:
        if not report['all_replica_model_optimizer_states_equal']:
            raise AssertionError('Model/optimizer replicas differ')
        if file_sha256(path/'after-121-model-optimizer.pt')!=report['final_checkpoint_sha256']:
            raise AssertionError('Final trained checkpoint file changed')
    if file_sha256(primary/'after-120-full-state.pt')!=a['checkpoint_sha256']:
        raise AssertionError('Recovery checkpoint file changed')
    before=torch.load(primary/'after-120-full-state.pt',map_location='cpu',mmap=True,weights_only=False)
    uninterrupted=torch.load(primary/'after-121-model-optimizer.pt',map_location='cpu',mmap=True,weights_only=False)
    replayed=torch.load(replay/'after-121-model-optimizer.pt',map_location='cpu',mmap=True,weights_only=False)
    for state in [before,uninterrupted,replayed]:finite_state(state)
    recovery=check_replay(before,uninterrupted,replayed)
    return {'arm':a['arm'],'precision':a['precision'],'accumulation':a['accumulation'],
        'parameters':sum(before['model'][n].numel() for group in before['optimizer']['recipe']['groups'] for n in group),
        'complete_state_on_load_exact':True,'per_rank_cpu_rng_on_load_exact':True,
        'all_replica_model_optimizer_states_equal':True,'replay':recovery,
        'passes_recovery_gate':recovery['passes_two_percent_gate'],
        'warm_real_targets':a['warm_real_targets'],'warm_seconds':a['warm_seconds'],
        'warm_targets_per_second_per_eight_chips':a['warm_targets_per_second'],
        'warm_model_matmuls':a['warm_model_matmuls'],
        'warm_optimizer_matmuls_all_replicas':a['warm_optimizer_matmuls_all_replicas'],
        'actual_optimizer_calls_per_replica_including_replay':122,
        'phase_training_real_targets_including_replay':a['phase_real_targets']+b['phase_real_targets'],
        'extra_initial_probe_model_matmuls':a['extra_initial_probe_model_matmuls'],
        'extra_initial_probe_real_targets':a['extra_initial_probe_real_targets']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    out=args.root/'parcae-low-depth-qualification-v1'
    public=Path('results/parcae_low_depth_qualification_20261024')
    preparation=json.loads((out/'plan.json').read_text())
    cpu=json.loads((out/'cpu-report.json').read_text())
    if cpu['status']!='COMPLETE_PARCAE_TRAINING_QUALIFICATION_PREPARATION':
        raise ValueError('Initial CPU/source/sampler preparation incomplete')
    plan_path=out/'native-plan.json'
    if not plan_path.exists():
        hashes={}
        for file in [out/'plan.json',args.root/'parcae-native-fp16-v1/plan.json']:
            for path,sha in json.loads(file.read_text())['source_sha256'].items():
                if path in hashes and hashes[path]!=sha:raise AssertionError('Preparation source freezes disagree')
                hashes[path]=sha
        files=['scripts/run_parcae_training_qualification.py','scripts/train_parcae_qualification.py',
            'scripts/audit_parcae_training_point.py','compute_trainium/parcae_work_accounting.py',
            'compute_trainium/parcae_recovery_check.py','docs/PARCAE_TRAINING_QUALIFICATION_20261024.md',
            'tests/test_parcae_training_candidate.py','tests/test_parcae_training_recovery.py',
            'tests/test_parcae_training_execution.py']
        hashes.update({f:file_sha256(f) for f in files})
        plan={'identity':'parcae-low-depth-qualification-v1','registered_unix':time.time(),
            'source_sha256':hashes,'preparation_plan_sha256':file_sha256(out/'plan.json'),
            'cpu_report_sha256':file_sha256(out/'cpu-report.json'),
            'latest_phase_start_unix':dt.datetime(2026,10,25,4,tzinfo=dt.timezone.utc).timestamp(),
            'native_phase_seconds_limit':1800,'execution_seconds_limit':5400,'cpu_audit_seconds_limit':1800,
            'world':32,'contexts':[1024],'accumulations':[8,16],'arms':['ordinary','fixed1','poisson2'],
            'warmup_updates':20,'measured_updates':100,'primary_optimizer_updates':121,'replay_updates':1,
            'gradient_relative_gate':.02,'nll_absolute_gate':.003,'replay_update_and_moment_relative_gate':.02,
            'scope':'Bounded complete training/profile qualification at actual140.8M ordinary and142.0M family inventories, shared backbone/release initialization, low-depth natural TRAIN only. No scientific fit, held-out access, terminal alpha or automatically authorized large budget. Precision set by earlier numerical gates, not quality scores.'}
        write_state(plan_path,plan);write_state(public/'native-plan.json',plan)
    plan=json.loads(plan_path.read_text())

    def frozen():
        if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):
            raise AssertionError('Training qualification source changed')
        if file_sha256(out/'plan.json')!=plan['preparation_plan_sha256'] or file_sha256(out/'cpu-report.json')!=plan['cpu_report_sha256']:
            raise AssertionError('Initial CPU/sampler preparation changed')
    frozen()
    if args.prepare_only:return
    if (out/'report.json').exists():raise FileExistsError('No silent training qualification rerun')
    waiting=time.time();started=None
    active={};cpu_active={};costs=[];cpu_costs=[];records=[];audit_records=[];audit_futures=[]
    lock=threading.Lock();failed=threading.Event()
    chosen=None
    cpu_pool=cf.ThreadPoolExecutor(2)

    def native(arm,precision,accumulation,half,mode,deadline):
        frozen()
        if failed.is_set():raise RuntimeError('Peer numerical failure; no phase starts')
        if time.time()>plan['latest_phase_start_unix']:raise TimeoutError('Native latest-start gate')
        identity=f'parcae-low-qual-v1-{precision}-a{accumulation}-{arm}-{mode}'
        container=create_worker(args.root,identity,half)
        with lock:active[identity]=container
        proc=None
        try:
            limit=min(deadline,time.time()+plan['native_phase_seconds_limit'])
            env=dict(os.environ,RESEARCH_ROOT=str(args.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                RESEARCH_RUN=identity,RESEARCH_OWNER=identity,RESEARCH_CONTAINER=container,
                RESEARCH_ENTRY_MODULE='scripts.train_parcae_qualification',RESEARCH_DDP_RANKS='32',
                NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,half*32+32))),
                RESEARCH_TIMEOUT_SECONDS=str(max(1,int(limit-time.time()))))
            command=[os.sys.executable,'-u','-m','scripts.supervise_isolated',
                '--corpus','/work/instance-nvme/natural-append-6500m-v2/source.json',
                '--packed-corpus','/work/instance-nvme/natural-append-6500m-v2/packed',
                '--preparation-root','/work/instance-nvme/parcae-low-depth-qualification-v1',
                '--arm',arm,'--precision',precision,'--accumulation',str(accumulation),'--mode',mode]
            if mode=='replay':
                parent=identity.removesuffix('-replay')+'-warm'
                command+=['--resume','/work/instance-nvme/runs/'+parent+'/after-120-full-state.pt']
            with (out/(identity+'.log')).open('x') as log:
                if time.time()>plan['latest_phase_start_unix']:raise TimeoutError('Container setup crossed latest start')
                proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if failed.is_set() or time.time()>limit:
                    raise RuntimeError('Training qualification paired stop or phase deadline')
                time.sleep(5)
            if proc.returncode:raise RuntimeError('Native complete training phase failed: '+identity)
            path=args.root/'runs'/identity
            report=json.loads((path/'report.json').read_text())
            if report['status']!='CAPTURED_PARCAE_TRAINING_QUALIFICATION':raise AssertionError('Native training phase did not qualify its local checks')
            return path
        finally:
            cancel_worker_container(container,identity,'training_qualification_phase_complete')
            if proc is not None and proc.poll() is None:
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.terminate()
            closure=args.root/'snapshots'/identity/'closure.json'
            with lock:
                active.pop(identity,None)
                costs.append({'identity':identity,**json.loads(closure.read_text())} if closure.exists() else {'identity':identity,'closure_missing':True})

    def trained_audit(key,primary,deadline):
        begin=time.time();proc=None
        audit_out=out/'trained-audits'/key
        audit_out.parent.mkdir(exist_ok=True)
        try:
            frozen()
            env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12')
            with (out/(key+'-cpu.log')).open('x') as log:
                proc=subprocess.Popen([os.sys.executable,'-u','-m','scripts.audit_parcae_training_point',
                    '--root',str(args.root),'--primary',str(primary),
                    '--upstream',str(args.root/'upstream-parcae-69284c1'),'--output',str(audit_out)],
                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:cpu_active[key]=proc
            limit=min(deadline,begin+plan['cpu_audit_seconds_limit'])
            while proc.poll() is None:
                if failed.is_set() or time.time()>limit:raise RuntimeError('Trained CPU audit paired stop/deadline')
                time.sleep(5)
            if proc.returncode:raise RuntimeError('Independent trained-point CPU process failed: '+key)
            result=json.loads((audit_out/'report.json').read_text())
            with lock:
                audit_records.append(result);write_state(public/'trained-audit-progress.json',{'records':audit_records})
            if result['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT':
                raise AssertionError('Independent trained-point numerical gate failed: '+key)
            return audit_out,result
        except BaseException:
            failed.set();raise
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            with lock:
                cpu_active.pop(key,None)
                cpu_costs.append({'identity':key,'started_unix':begin,'finished_unix':time.time(),
                    'wall_seconds':time.time()-begin,'assigned_chips':0,'exit_code':proc.returncode if proc else None})

    def worker(role,half,precisions,deadline):
        try:
            for precision in precisions:
                configurations=([('ordinary',16),('fixed1',8),('fixed1',16)] if role=='controls' else
                                [('poisson2',8),('poisson2',16)])
                for arm,accumulation in configurations:
                    key=f'{precision}-a{accumulation}-{arm}'
                    primary=native(arm,precision,accumulation,half,'warm',deadline)
                    future=cpu_pool.submit(trained_audit,key,primary,deadline)
                    with lock:audit_futures.append((key,primary,future))
                    replay=native(arm,precision,accumulation,half,'replay',deadline)
                    result=check_native_pair(primary,replay)
                    with lock:
                        records.append(result);write_state(public/'native-progress.json',{'records':records})
                    if not result['passes_recovery_gate']:raise AssertionError('Next-update numerical replay gate failed: '+key)
        except BaseException:
            failed.set();raise

    try:
        while True:
            frozen()
            paths=[args.root/'parcae-native-global-v1/native-report.json',
                args.root/'parcae-optimizer-global-v1/attempt2/native-report.json',
                args.root/'parcae-native-fp16-v1/report.json']
            states=[json.loads(p.read_text()) if p.exists() else None for p in paths]
            chosen=launch_policy(*states,bool(managed_running()) if all(states) else True,
                time.time(),plan['latest_phase_start_unix'])
            if chosen:break
            write_state(public/'native-status.json',{'status':'WAITING_FOR_PRIOR_NUMERICAL_GATES','assigned_chips':0,'observed_unix':time.time()})
            time.sleep(30)
        started=time.time();deadline=started+plan['execution_seconds_limit']
        write_state(public/'precision-selection.json',{'precisions':chosen,
            'criterion':'FP32 gradient and optimizer/recovery must pass. Add FP16 only after its native gradient pass; otherwise keep the qualified FP32 path. No held-out quality or runtime-based selection.'})
        write_state(public/'native-status.json',{'status':'RUNNING_BOUNDED_TRAINING_PROFILE_QUALIFICATION','maximum_chips':16,'started_unix':started})
        with cf.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(worker,role,i,chosen,deadline) for i,role in enumerate(['controls','recurrence'])]
            for future in cf.as_completed(futures):future.result()
        for key,primary,future in audit_futures:
            audit_path,audit_result=future.result(timeout=max(1,deadline-time.time()))
            vector=audit_path/'cpu-global.pt'
            if file_sha256(vector)!=audit_result['cpu_global_gradient_sha256']:
                raise AssertionError('Independent trained CPU vector changed')
            expected=torch.load(vector,map_location='cpu',mmap=True,weights_only=False)
            replay=args.root/'runs'/primary.name.removesuffix('-warm')
            replay=replay.with_name(replay.name+'-replay')
            actual=torch.load(replay/'updated-global-native.pt',map_location='cpu',mmap=True,weights_only=False)
            if expected['real_targets']!=actual['real_targets'] or expected['source_cursor']!=actual['source_cursor']:
                raise AssertionError('Resumed global objective exposure/cursor differs: '+key)
            delta=gradient_comparison(expected['gradients'],actual['gradients'])
            difference=actual['loss']-expected['loss']
            if not delta['passes_two_percent_gate'] or abs(difference)>.003:
                raise AssertionError('Resumed trained global gradient differs from independent CPU: '+key)
            row=next(r for r in records if f"{r['precision']}-a{r['accumulation']}-{r['arm']}"==key)
            row['independent_trained_gradient']=audit_result
            row['resumed_gradient_vs_independent_cpu']={**delta,'native_minus_cpu_nll':difference}
        frozen()
        if time.time()>deadline:raise TimeoutError('Full training qualification execution window expired')
        result={'status':'PASS_COMPLETE_PARCAE_TRAINING_QUALIFICATION','records':records,'precisions':chosen}
    except BaseException as error:
        failed.set()
        result={'status':'STOPPED_COMPLETE_PARCAE_TRAINING_QUALIFICATION','records':records,'error':str(error),'precisions':chosen}
        raise
    finally:
        failed.set()
        for identity,container in list(active.items()):
            try:cancel_worker_container(container,identity,'training_qualification_stop')
            except BaseException:pass
        for proc in list(cpu_active.values()):
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        cpu_pool.shutdown(wait=True,cancel_futures=True)
        now=time.time()
        result.update(waiting_seconds=(started or now)-waiting,active_execution_seconds=now-started if started else 0.,
            test_accessed=False,research_optimizer_updates=0,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'native-status.json',{'status':result['status'],'assigned_chips':0,'observed_unix':now})
        write_state(public/'execution-costs.json',{'status':result['status'],'native_attempts':costs,'cpu_audits':cpu_costs,
            'waiting_seconds':result['waiting_seconds'],'active_execution_seconds':result['active_execution_seconds'],
            'scope':'All cold/native updates, source checks, checkpoints/replays, CPU tails and failed attempts are included. Warm100 excludes initial compile/probe and final audits. Waiting assigns zero chips. Do not add concurrent wall intervals as sequential reservation time. Qualification source is intentionally reused across configurations; zero scientific exposure or claim.'})


if __name__=='__main__':main()
