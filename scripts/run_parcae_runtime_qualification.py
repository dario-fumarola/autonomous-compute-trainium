"""Three full training/recovery profiles with100 changing-schedule updates."""
import argparse
import concurrent.futures as cf
import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running,cancel_worker_container
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.owned_native_compact import create_compact_worker,reclaim_host
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_parcae_resumed_training_qualification import check_native_pair


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root/'parcae-runtime-scalars-qualification-v1';public=Path('results/parcae_runtime_scalars_20261024')
    if out.exists():raise FileExistsError('Preserve previous runtime-scalar qualification')
    previous=Path('results/parcae_low_depth_qualification_20261024/attempt4/report.json')
    if json.loads(previous.read_text())['status']!='PASS_COMPLETE_PARCAE_TRAINING_QUALIFICATION':raise ValueError('Underlying operators must already be qualified')
    plan_sources=dict(json.loads(Path('results/parcae_low_depth_qualification_20261024/attempt4/native-plan.json').read_text())['source_sha256'])
    files=['scripts/run_parcae_runtime_qualification.py','scripts/train_parcae_runtime_qualification.py',
        'compute_trainium/parcae_runtime_scalars.py','tests/test_parcae_runtime_scalars.py',
        'docs/PARCAE_RUNTIME_SCALARS_20261024.md','compute_trainium/native_comm_ports.py']
    plan_sources.update({f:file_sha256(f) for f in files})
    identities=[f'parcae-runtime-scalars-v1-{arm}-{mode}' for arm in ['ordinary','poisson2','fixed1'] for mode in ['warm','replay']]
    plan={'identity':'parcae-runtime-scalars-qualification-v1','registered_unix':time.time(),
        'source_sha256':plan_sources,'underlying_qualification_sha256':file_sha256(previous),
        'root_comm_ports':reserve_ports(a.root,identities),'seconds_limit':2700,'phase_seconds_limit':600,
        'cpu_seconds_limit':1200,'arms':['ordinary','poisson2','fixed1'],'precision':'fp16','accumulation':16,
        'scope':'Same FP32-polar Muon math/moments/grouping,packed runtime FP32 scalar copy replaces static fills. Three complete20+100+1-update profiles and fresh-process replay,independent first-update/global-trained gradients,100 unique LR/decay values. Qualification-only naturalTRAIN,zero heldout/researchupdates. All failures and original costs retained.'}
    out.mkdir();public.mkdir(parents=True,exist_ok=False);write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    started=time.time();deadline=started+plan['seconds_limit'];failed=threading.Event();lock=threading.Lock()
    tasks=queue.Queue()
    for arm in plan['arms']:tasks.put(arm)
    active={};cpu_active={};records=[];costs=[];cpu_costs=[];audit_futures=[];cpu_pool=cf.ThreadPoolExecutor(2)
    def frozen():
        if any(file_sha256(f)!=sha for f,sha in plan_sources.items()):raise AssertionError('Runtime-scalar qualification source changed')
    def native(arm,half,mode):
        frozen();reclaim_host();identity=f'parcae-runtime-scalars-v1-{arm}-{mode}'
        container=create_compact_worker(a.root,identity,half);process=None
        with lock:active[identity]=container
        try:
            limit=min(deadline,time.time()+plan['phase_seconds_limit'])
            env=dict(os.environ,RESEARCH_ROOT=str(a.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                RESEARCH_RUN=identity,RESEARCH_OWNER=identity,RESEARCH_CONTAINER=container,
                RESEARCH_ENTRY_MODULE='scripts.train_parcae_runtime_qualification',RESEARCH_DDP_RANKS='32',
                NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,half*32+32))),
                RESEARCH_TIMEOUT_SECONDS=str(max(1,int(limit-time.time()))))
            args=[os.sys.executable,'-u','-m','scripts.supervise_isolated',
                '--corpus','/work/instance-nvme/natural-append-6500m-v2/source.json',
                '--packed-corpus','/work/instance-nvme/natural-append-6500m-v2/packed',
                '--preparation-root','/work/instance-nvme/parcae-low-depth-qualification-v1',
                '--arm',arm,'--precision','fp16','--accumulation','16','--mode',mode,
                '--comm-port',str(plan['root_comm_ports'][identity])]
            if mode=='replay':args+=['--resume',f'/work/instance-nvme/runs/parcae-runtime-scalars-v1-{arm}-warm/after-120-full-state.pt']
            with (out/(identity+'.log')).open('x') as log:process=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if failed.is_set() or time.time()>limit:raise RuntimeError('Runtime qualification paired stop/deadline')
                time.sleep(5)
            if process.returncode:raise RuntimeError('Runtime-scalar native phase failed: '+identity)
            path=a.root/'runs'/identity;row=json.loads((path/'report.json').read_text())
            if row.get('scalar_input_policy')!='runtime_host_f32_tensor_copy':raise AssertionError('Scalar path not qualified')
            return path
        finally:
            cancel_worker_container(container,identity,'runtime_scalar_qualification_closed')
            if process and process.poll() is None:
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.terminate()
            closure=a.root/'snapshots'/identity/'closure.json'
            with lock:
                active.pop(identity,None);costs.append({'identity':identity,**json.loads(closure.read_text())} if closure.exists() else {'identity':identity,'closure_missing':True})
    def audit(arm,primary):
        frozen();begin=time.time();process=None;target=out/'cpu'/arm;target.parent.mkdir(exist_ok=True)
        try:
            env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12')
            args=[os.sys.executable,'-u','-m','scripts.audit_parcae_precise_training_point','--root',str(a.root),
                '--primary',str(primary),'--upstream',str(a.root/'upstream-parcae-69284c1'),'--output',str(target)]
            with (out/(arm+'-cpu.log')).open('x') as log:process=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:cpu_active[arm]=process
            while process.poll() is None:
                if failed.is_set() or time.time()>min(deadline,begin+plan['cpu_seconds_limit']):raise RuntimeError('Runtime CPU audit deadline')
                time.sleep(5)
            if process.returncode:raise RuntimeError('Runtime-scalar CPU audit failed: '+arm)
            row=json.loads((target/'report.json').read_text())
            if row['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT':raise AssertionError('Runtime trained gradient gate failed')
            return arm,row
        except BaseException:failed.set();raise
        finally:
            if process and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            with lock:
                cpu_active.pop(arm,None);cpu_costs.append({'arm':arm,'assigned_chips':0,'wall_seconds':time.time()-begin,
                    'started_unix':begin,'finished_unix':time.time(),'exit_code':process.returncode if process else None})
    def worker(half):
        try:
            while not failed.is_set():
                try:arm=tasks.get_nowait()
                except queue.Empty:return
                primary=native(arm,half,'warm');future=cpu_pool.submit(audit,arm,primary)
                with lock:audit_futures.append(future)
                replay=native(arm,half,'replay');row=check_native_pair(primary,replay);reclaim_host()
                if not row['passes_recovery_gate']:raise AssertionError('Runtime scalar recovery failed')
                with lock:records.append(row);write_state(public/'native-progress.json',{'records':records})
        except BaseException:failed.set();raise
    result={'status':'STOPPED_PARCAE_RUNTIME_SCALAR_QUALIFICATION'}
    try:
        while managed_running():
            if time.time()>deadline:raise TimeoutError('Runtime-scalar admission deadline')
            time.sleep(5)
        frozen();write_state(public/'status.json',{'status':'RUNNING_CHANGING_SCHEDULE_PROFILES','maximum_chips':16})
        with cf.ThreadPoolExecutor(2) as pool:
            for future in cf.as_completed([pool.submit(worker,half) for half in [0,1]]):future.result()
        for future in audit_futures:
            arm,row=future.result(timeout=max(1,deadline-time.time()))
            next(r for r in records if r['arm']==arm)['independent_trained_gradient']=row
        frozen()
        if len(records)!=3 or time.time()>deadline:raise AssertionError('Runtime qualification incomplete')
        result={'status':'PASS_COMPLETE_PARCAE_RUNTIME_SCALAR_QUALIFICATION','records':records}
    except BaseException as error:failed.set();result.update(error=str(error),records=records);raise
    finally:
        failed.set()
        for identity,container in list(active.items()):
            try:cancel_worker_container(container,identity,'runtime_scalar_paired_stop')
            except BaseException:pass
        for process in list(cpu_active.values()):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        cpu_pool.shutdown(wait=True,cancel_futures=True)
        result.update(wall_seconds=time.time()-started,research_optimizer_updates=0,test_accessed=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0})
        write_state(public/'execution-costs.json',{'native_attempts':costs,'cpu_audits':cpu_costs,'wall_seconds':time.time()-started,'scope':plan['scope']})


if __name__=='__main__':main()
