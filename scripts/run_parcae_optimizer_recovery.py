"""Replay only the passing FP32-polar primary states with smaller host caps."""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import torch

from compute_trainium.execution_isolation import managed_running,cancel_worker_container
from compute_trainium.owned_native_compact import create_compact_worker,reclaim_host
from compute_trainium.parcae_update_qualification import compare_update
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_parcae_optimizer_precise_native import step_comparison,step_passes


def replay_comparison(cpu,primary,replay,arm):
    expected2=torch.load(cpu/arm/'cpu-step-2.pt',map_location='cpu',mmap=True,weights_only=False)
    expected3=torch.load(cpu/arm/'cpu-step-3.pt',map_location='cpu',mmap=True,weights_only=False)
    actual2=torch.load(primary/'native-step-2.pt',map_location='cpu',mmap=True,weights_only=False)
    actual3=torch.load(primary/'native-step-3.pt',map_location='cpu',mmap=True,weights_only=False)
    resumed=torch.load(replay/'native-step-3.pt',map_location='cpu',mmap=True,weights_only=False)
    delta=step_comparison(expected2['model'],expected3,actual2['model'],resumed)
    continuation=compare_update(actual2['model'],actual3['model'],actual2['model'],resumed['model'])
    return {'replay_vs_cpu':delta,'replay_vs_uninterrupted_update':continuation,
            'passes_registered_gates':step_passes(delta) and continuation['relative_l2']<=2e-4}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    out=args.root/'parcae-optimizer-recovery-v2';out.mkdir(parents=True,exist_ok=True)
    public=Path('results/parcae_optimizer_precise_20261024/recovery-v2');public.mkdir(parents=True,exist_ok=True)
    previous=args.root/'parcae-optimizer-precise-v1/native'
    cpu=args.root/'parcae-optimizer-precise-v1/cpu'
    gradient=args.root/'parcae-native-global-v1'
    plan_file=out/'native-plan.json'
    if not plan_file.exists():
        prior=json.loads((previous/'native-plan.json').read_text())
        cpu_proof=json.loads((cpu/'report.json').read_text())
        if cpu_proof['status']!='PASS_ACTUAL_SHAPE_CPU_FP32_POLAR_OPTIMIZER_VECTORS':
            raise AssertionError('Independent FP32-polar CPU vectors required')
        hashes=dict(prior['source_sha256'])
        for p in ['scripts/run_parcae_optimizer_recovery.py','compute_trainium/owned_native_compact.py',
                  'tests/test_compact_native_worker.py','tests/test_parcae_recovery_execution.py',
                  'docs/PARCAE_RECOVERY_ADMISSION_20261024.md']:
            hashes[p]=file_sha256(p)
        artifacts={str(cpu/p):file_sha256(cpu/p) for p in ['plan.json','report.json']}
        for arm in ['ordinary','recurrent']:
            primary=args.root/'runs'/f'parcae-native-opt-precise-v1-{arm}-primary'
            proof=json.loads((primary/'report.json').read_text())
            check_file=Path('results/parcae_optimizer_precise_20261024/native')/(arm+'-primary.json')
            checks=json.loads(check_file.read_text())
            if not proof['all_complete_replica_states_equal'] or not all(step_passes(r) for r in checks['updates']):
                raise AssertionError('Cannot reuse a failed primary capture')
            if file_sha256(primary/'after-two-full-state.pt')!=proof['recovery_checkpoint_sha256']:
                raise AssertionError('Original full recovery checkpoint changed')
            artifacts[str(check_file)]=file_sha256(check_file)
            for file in ['report.json','native-step-1.pt','native-step-2.pt','native-step-3.pt','after-two-full-state.pt']:
                artifacts[str(primary/file)]=file_sha256(primary/file)
            for step in range(1,4):
                p=cpu/arm/f'cpu-step-{step}.pt'
                sha=file_sha256(p)
                expected=next(r for r in cpu_proof['records'] if r['arm']==arm)['vectors'][step-1]['checkpoint_sha256']
                if sha!=expected:raise AssertionError('Independent CPU update vector changed')
                artifacts[str(p)]=sha
        plan={'identity':'parcae-optimizer-recovery-v2','registered_unix':time.time(),
              'source_sha256':hashes,'input_sha256':artifacts,'polar_dtype':'fp32','execution':'eager_muon_fused_adamw',
              'world':32,'worker_host_cap_gib':128,'aggregate_workload_cap_gib':256,
              'seconds_limit':1800,'latest_start_unix':dt.datetime(2026,10,25,2,tzinfo=dt.timezone.utc).timestamp(),
              'update_relative_gate':.02,'moment_relative_gate':.02,'replay_relative_gate':2e-4,
              'new_replay_calls_per_replica_per_arm':1,'retained_primary_calls_per_replica_per_arm':3,
              'test_accessed':False,'scope':'Reuse exact passing primary states/updates; new fresh-process step3 replay only. Smaller owned caps preserve512GiB reserve. No new scientific update, source exposure, held-out scoring or changed optimizer arithmetic.'}
        write_state(plan_file,plan);write_state(public/'native-plan.json',plan)
    plan=json.loads(plan_file.read_text())
    def frozen(*,inputs=False):
        checks=dict(plan['source_sha256'])
        if inputs:checks.update(plan['input_sha256'])
        if any(file_sha256(p)!=sha for p,sha in checks.items()):raise AssertionError('Frozen recovery source/input changed')
    frozen(inputs=True)
    if args.prepare_only:return
    if (out/'native-report.json').exists():raise FileExistsError('No silent recovery rerun')
    waiting=time.time();started=None;active={};costs=[];records=[]
    lock=threading.Lock();failed=threading.Event()
    def run(arm,half,deadline):
        identity='parcae-opt-recovery-v2-'+arm
        container=None;process=None
        try:
            frozen();reclaim_host()
            if time.time()>plan['latest_start_unix']:raise TimeoutError('Recovery latest-start deadline')
            container=create_compact_worker(args.root,identity,half)
            with lock:active[identity]=container
            primary=args.root/'runs'/f'parcae-native-opt-precise-v1-{arm}-primary'
            env=dict(os.environ,RESEARCH_ROOT=str(args.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                     RESEARCH_RUN=identity,RESEARCH_OWNER=identity,RESEARCH_CONTAINER=container,
                     RESEARCH_ENTRY_MODULE='scripts.capture_parcae_optimizer_precise',RESEARCH_DDP_RANKS='32',
                     NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,half*32+32))),
                     RESEARCH_TIMEOUT_SECONDS=str(max(1,int(deadline-time.time()))))
            command=[os.sys.executable,'-u','-m','scripts.supervise_isolated',
                '--optimizer-root','/work/instance-nvme/parcae-optimizer-precise-v1/cpu',
                '--gradient-root','/work/instance-nvme/parcae-native-global-v1','--arm',arm,'--mode','replay',
                '--primary-root','/work/instance-nvme/runs/'+primary.name]
            with (out/(arm+'.log')).open('x') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if failed.is_set() or time.time()>deadline:raise RuntimeError('Paired recovery stop/deadline')
                time.sleep(5)
            if process.returncode:raise RuntimeError('Native recovery worker failed: '+arm)
            replay=args.root/'runs'/identity
            proof=json.loads((replay/'report.json').read_text())
            if not proof['complete_state_on_load_exact'] or not proof['all_complete_replica_states_equal'] or proof['actual_optimizer_calls_per_replica']!=1:
                raise AssertionError('Complete recovery/replica/count gate failed')
            # Release native resources before CPU comparison allocates scratch.
            cancel_worker_container(container,identity,'replay_native_complete')
            with lock:active.pop(identity,None)
            result={'arm':arm,**replay_comparison(cpu,primary,replay,arm),
                    'complete_state_on_load_exact':True,'all_complete_replica_states_equal':True,
                    'new_optimizer_calls_per_replica':1,'retained_primary_calls_per_replica':3}
            with lock:records.append(result);write_state(public/'progress.json',{'records':records})
            if not result['passes_registered_gates']:raise AssertionError('Recovery numerical gate failed: '+arm)
        except BaseException:
            failed.set();raise
        finally:
            if container:
                cancel_worker_container(container,identity,'recovery_phase_complete')
                with lock:active.pop(identity,None)
            if process and process.poll() is None:
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.terminate()
            closure=args.root/'snapshots'/identity/'closure.json'
            with lock:costs.append({'identity':identity,**json.loads(closure.read_text())} if closure.exists() else {'identity':identity,'closure_missing':True})
            reclaim_host()
    try:
        while managed_running():
            if time.time()>plan['latest_start_unix']:raise TimeoutError('Recovery admission deadline')
            write_state(public/'status.json',{'status':'WAITING_FOR_NATIVE_OWNERS','assigned_chips':0,'observed_unix':time.time()})
            time.sleep(10)
        if json.loads((gradient/'native-report.json').read_text())['status']!='PASS_PAIRED_NATIVE_PARCAE_STATIC_STAGE_GLOBAL_GRADIENTS':
            raise AssertionError('Prior native full-gradient gate required')
        started=time.time();deadline=started+plan['seconds_limit']
        write_state(public/'status.json',{'status':'RUNNING_PAIRED_FP32_POLAR_RECOVERY','maximum_chips':16,'started_unix':started})
        with cf.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(run,arm,half,deadline) for half,arm in enumerate(['ordinary','recurrent'])]
            for future in cf.as_completed(futures):future.result()
        frozen(inputs=True)
        if time.time()>deadline:raise TimeoutError('Recovery final comparisons exceeded execution cap')
        result={'status':'PASS_PAIRED_NATIVE_PARCAE_FP32_POLAR_OPTIMIZER_AND_RECOVERY','records':records}
    except BaseException as e:
        failed.set();result={'status':'STOPPED_FP32_POLAR_OPTIMIZER_RECOVERY','records':records,'error':str(e)}
        raise
    finally:
        failed.set()
        for identity,container in list(active.items()):cancel_worker_container(container,identity,'recovery_stop')
        now=time.time();result.update(active_execution_seconds=now-started if started else 0.,waiting_seconds=(started or now)-waiting,
            research_optimizer_updates=0,test_accessed=False,scope=plan['scope'])
        write_state(out/'native-report.json',result);write_state(public/'native-report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0,'observed_unix':now})
        write_state(public/'execution-costs.json',{'attempts':costs,'active_execution_seconds':result['active_execution_seconds'],
            'waiting_seconds':result['waiting_seconds'],'scope':'New recovery work only; retain primary and failed/canceled replay campaign receipts separately. Overlapping intervals are not sequential reservation time.'})


if __name__=='__main__':main()
