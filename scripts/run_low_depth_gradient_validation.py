"""Finite owned queue: capture full gradients then independently check on CPU."""
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


def register(root,identity,parent_file,points_file):
    out=root/identity;public=Path('results')/identity
    if out.exists():raise FileExistsError('No silent gradient-validation plan replacement')
    parent=json.loads(parent_file.read_text());points=json.loads(points_file.read_text())
    if parent['purpose'] not in ['low_depth_driver_integration_qualification','natural_low_depth_scientific_pilot'] or not 1<=len(points)<=3:
        raise ValueError('Finite registered low-depth checkpoint set required')
    for key,row in points.items():
        if not key.replace('-','').isalnum() or row['case_id'] not in parent['cases']:
            raise ValueError('Explicit safe checkpoint/case identity required')
        if file_sha256(root/row['checkpoint'])!=row['checkpoint_sha256']:raise AssertionError('Checkpoint identity changed')
    files=['scripts/run_low_depth_gradient_validation.py','scripts/capture_low_depth_gradient.py',
        'scripts/audit_low_depth_gradient.py','compute_trainium/low_depth_gradient_probe.py',
        'tests/test_low_depth_gradient_probe.py','docs/LOW_DEPTH_MATURE_GRADIENT_POLICY_20261024.md']
    sources={**parent['source_sha256'],**{f:file_sha256(f) for f in files}}
    # Parent launch/controller sources remain frozen too; new helpers cannot
    # invalidate the parent manifest merely by appearing in the directory.
    if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Registered parent source changed')
    plan={'identity':identity,'registered_unix':time.time(),'purpose':'zero_update_low_depth_gradient_validation',
        'parent_plan':str(parent_file.relative_to(root)),'parent_plan_sha256':file_sha256(parent_file),
        'points_manifest_sha256':file_sha256(points_file),'points':points,'source_sha256':sources,
        'worker_source_sha256':{f:sha for f,sha in sources.items() if f.startswith(('scripts/','compute_trainium/','configs/'))},
        'root_comm_ports':reserve_ports(root,[identity+'-'+key for key in points]),
        'seconds_limit':2700,'phase_seconds_limit':600,'cpu_seconds_limit':1200,
        'test_accessed':False,'research_optimizer_updates':0,
        'scope':'Frozen checkpoints,next complete512-window natural TRAIN objective,zero updates. Captured exact supplied hidden states/depths, independent upstream FP32 overall/named-family2% gradient and0.003-nat loss gates. All diagnostic forward/backward work and CPU/native time charged separately.'}
    out.mkdir();public.mkdir(parents=True,exist_ok=False)
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--identity',required=True);p.add_argument('--parent-plan',type=Path)
    p.add_argument('--points-manifest',type=Path);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if not a.identity.replace('-','').isalnum():raise ValueError('Safe explicit execution identity required')
    out=a.root/a.identity;public=Path('results')/a.identity;plan_file=out/'plan.json'
    if plan_file.exists():
        plan=json.loads(plan_file.read_text())
        if a.parent_plan or a.points_manifest:raise ValueError('Existing frozen plan does not accept new inputs')
    else:
        if not a.parent_plan or not a.points_manifest:raise ValueError('Initial registration needs exact parent and checkpoint manifest')
        plan=register(a.root,a.identity,a.parent_plan,a.points_manifest)
    if a.prepare_only:return
    if (out/'report.json').exists() or (out/'started.json').exists():raise FileExistsError('Preserve an executed probe; register explicit recovery separately')
    plan_sha=file_sha256(plan_file);started=time.time();deadline=started+plan['seconds_limit']
    write_state(out/'started.json',{'started_unix':started,'plan_sha256':plan_sha})
    tasks=queue.Queue()
    for key in plan['points']:tasks.put(key)
    lock=threading.Lock();failed=threading.Event();active={};cpu_active={};records=[];costs=[];cpu_costs=[]
    def frozen():
        if file_sha256(plan_file)!=plan_sha or any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):
            raise AssertionError('Frozen gradient validation changed')
    def capture(key,half):
        frozen();reclaim_host();identity=a.identity+'-'+key
        container=create_compact_worker(a.root,identity,half);process=None;begin=time.time()
        with lock:active[identity]=container
        try:
            limit=min(deadline,begin+plan['phase_seconds_limit'])
            env=dict(os.environ,RESEARCH_ROOT=str(a.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                RESEARCH_RUN=identity,RESEARCH_OWNER=identity,RESEARCH_CONTAINER=container,
                RESEARCH_ENTRY_MODULE='scripts.capture_low_depth_gradient',RESEARCH_DDP_RANKS='32',
                NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,half*32+32))),
                RESEARCH_TIMEOUT_SECONDS=str(max(1,int(limit-time.time()))))
            args=[os.sys.executable,'-u','-m','scripts.supervise_isolated','--root','/work/instance-nvme',
                '--plan','/work/instance-nvme/'+a.identity+'/plan.json','--plan-sha',plan_sha,
                '--point',key,'--comm-port',str(plan['root_comm_ports'][identity])]
            with (out/(key+'-native.log')).open('x') as log:process=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if failed.is_set() or time.time()>limit:raise RuntimeError('Gradient capture paired stop/deadline')
                time.sleep(5)
            if process.returncode:raise RuntimeError('Gradient capture failed: '+key)
            report=json.loads((a.root/'runs'/identity/'report.json').read_text())
            if report['status']!='CAPTURED_ZERO_UPDATE_LOW_DEPTH_GLOBAL_GRADIENT':raise AssertionError('Capture incomplete')
            return identity,report
        finally:
            cancel_worker_container(container,identity,'gradient_capture_complete')
            if process and process.poll() is None:
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.terminate()
            closure=a.root/'snapshots'/identity/'closure.json'
            with lock:
                active.pop(identity,None)
                costs.append({'identity':identity,**json.loads(closure.read_text())} if closure.exists() else {'identity':identity,'closure_missing':True})
    def audit(key,identity,native):
        frozen();begin=time.time();process=None;target=out/'cpu'/key
        try:
            target.parent.mkdir(exist_ok=True)
            args=[os.sys.executable,'-u','-m','scripts.audit_low_depth_gradient','--root',str(a.root),
                '--plan',str(plan_file),'--plan-sha',plan_sha,'--point',key,'--capture',str(a.root/'runs'/identity),
                '--output',str(target),'--upstream',str(a.root/'upstream-parcae-69284c1'),
                '--seconds-limit',str(plan['cpu_seconds_limit'])]
            env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12')
            with (out/(key+'-cpu.log')).open('x') as log:
                process=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:cpu_active[key]=process
            while process.poll() is None:
                if failed.is_set() or time.time()>min(deadline,begin+plan['cpu_seconds_limit']):raise RuntimeError('Independent gradient audit paired stop/deadline')
                time.sleep(5)
            if process.returncode:raise RuntimeError('Independent gradient audit failed: '+key)
            row=json.loads((target/'report.json').read_text())
            if row['status']!='PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT':raise AssertionError('Unqualified mature gradient')
            with lock:
                records.append({'native':{k:v for k,v in native.items() if k not in ['probe_rows','noise_sha256_by_rank']},'independent_cpu':row})
                write_state(public/'progress.json',{'records':records})
        except BaseException:failed.set();raise
        finally:
            if process and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            with lock:
                cpu_active.pop(key,None);cpu_costs.append({'point':key,'assigned_chips':0,'wall_seconds':time.time()-begin,
                    'started_unix':begin,'finished_unix':time.time(),'exit_code':process.returncode if process else None})
    cpu_pool=cf.ThreadPoolExecutor(3);cpu_futures=[]
    def worker(half):
        try:
            while not failed.is_set():
                try:key=tasks.get_nowait()
                except queue.Empty:return
                identity,native=capture(key,half)
                with lock:cpu_futures.append(cpu_pool.submit(audit,key,identity,native))
        except BaseException:failed.set();raise
    result={'status':'STOPPED_LOW_DEPTH_GRADIENT_VALIDATION'}
    try:
        frozen()
        while managed_running():
            if time.time()>deadline:raise TimeoutError('Gradient-validation admission deadline')
            write_state(public/'status.json',{'status':'WAITING_FOR_ACTIVE_OWNERS','assigned_chips':0});time.sleep(5)
        write_state(public/'status.json',{'status':'RUNNING_FULL_GLOBAL_GRADIENT_VALIDATION','maximum_chips':16})
        with cf.ThreadPoolExecutor(2) as pool:
            for future in cf.as_completed([pool.submit(worker,half) for half in [0,1]]):future.result()
        for future in cpu_futures:future.result(timeout=max(1,deadline-time.time()))
        frozen();result={'status':'PASS_LOW_DEPTH_GRADIENT_VALIDATION','records':records}
    except BaseException as error:failed.set();result.update(error=str(error),records=records);raise
    finally:
        failed.set()
        for identity,container in list(active.items()):
            try:cancel_worker_container(container,identity,'gradient_validation_stop')
            except BaseException:pass
        for process in list(cpu_active.values()):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        cpu_pool.shutdown(wait=True,cancel_futures=True)
        result.update(wall_seconds=time.time()-started,test_accessed=False,research_optimizer_updates=0,plan_sha256=plan_sha)
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0})
        write_state(public/'execution-costs.json',{'native_attempts':costs,'cpu_audits':cpu_costs,
            'wall_seconds':time.time()-started,'scope':plan['scope']})


if __name__=='__main__':main()
