"""Owned finite qualification queue,admitted only after predecessor hardware."""
import argparse
import concurrent.futures as cf
import fcntl
import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running,cancel_worker_container
from compute_trainium.future_span_admission import predecessor_execution,completed_native_receipts
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.owned_native_compact import create_compact_worker,reclaim_host
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def register(root):
    identity='future-span-qualification-v1';out=root/identity
    public=Path('results/future_span_qualification_20261025/native-v1')
    preparation=root/'future-span-qualification-preparation-v1'
    receipt=json.loads((preparation/'report.json').read_text())
    if receipt['status']!='COMPLETE_CPU_FUTURE_SPAN_QUALIFICATION_PREPARATION':raise ValueError('Complete CPU preparation required')
    manifest=json.loads((preparation/'manifest.json').read_text())
    if file_sha256(preparation/'manifest.json')!=receipt['manifest_sha256']:raise AssertionError('CPU preparation changed')
    files=['scripts/run_future_span_qualification.py','scripts/qualify_future_span_native.py',
        'scripts/audit_future_span_qualification.py','compute_trainium/future_span_training.py',
        'compute_trainium/future_span_reference_optimizer.py','compute_trainium/future_span_admission.py',
        'compute_trainium/parcae_runtime_scalars.py','compute_trainium/parcae_optimizer_precise.py',
        'compute_trainium/parcae_optimizer_alignment.py','compute_trainium/parcae_recovery_check.py',
        'compute_trainium/parcae_training_candidate.py','compute_trainium/parcae_work_accounting.py',
        'compute_trainium/parcae_update_qualification.py','compute_trainium/stable_recurrence_candidate.py',
        'compute_trainium/depth_mixture.py','compute_trainium/metrics.py',
        'scripts/audit_parcae_precise_training_point.py','scripts/qualify_parcae_candidate_cpu.py',
        'scripts/run_parcae_optimizer_native.py','scripts/qualify_native.py',
        'third_party/parcae_reference/muon_adamw.py','docs/FUTURE_SPAN_NATIVE_QUALIFICATION_20261025.md']
    sources={**manifest['source_sha256'],**{p:file_sha256(p) for p in files}}
    if any(file_sha256(p)!=sha for p,sha in sources.items()):raise AssertionError('Original CPU preparation source changed')
    parent=root/'parcae-low-depth-pilot-v1/main-plan-recovery-v2.json'
    predecessor=json.loads(parent.read_text())
    cases={row['identity']:row for row in manifest['cases']}
    names=[identity+'-'+key+'-'+mode for key in cases for mode in ['warm','replay']]
    plan={'identity':identity,'registered_unix':time.time(),'purpose':'finite_future_span_training_qualification',
        'qualification_only':True,'preparation':str(preparation.relative_to(root)),
        'preparation_manifest_sha256':file_sha256(preparation/'manifest.json'),
        'predecessor_plan':str(parent.relative_to(root)),'predecessor_plan_sha256':file_sha256(parent),
        'predecessor_deadline_unix':predecessor['registered_unix']+predecessor['seconds_limit'],
        'cases':cases,'source_sha256':sources,
        'worker_source_sha256':{p:sha for p,sha in sources.items() if p.startswith(('compute_trainium/','scripts/','third_party/'))},
        'root_comm_ports':reserve_ports(root,names),'execution_seconds_limit':5400,
        'native_phase_seconds_limit':900,'cpu_phase_seconds_limit':1200,
        'scope':'Finite four-cell full-objective training/recovery qualification only. Admission requires all9 '
        'predecessor fits and9 numerical gates with successful native closures. No score-based selection or '
        'automatic scientific launch. All actual qualification updates,failures,CPU audits and reservation '
        'time are charged. Original admitted deadline and saved file hashes persist across controller resume.'}
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    return out,public,plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.root/'future-span-qualification-v1'
    public=Path('results/future_span_qualification_20261025/native-v1');plan_file=out/'plan.json'
    if plan_file.exists():plan=json.loads(plan_file.read_text())
    else:out,public,plan=register(args.root)
    if args.prepare_only:return
    lock_file=(out/'controller.lock').open('a')
    fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'report.json').exists():raise FileExistsError('Preserve a terminal qualification;new recovery registration required')
    started_file=out/'started.json';plan_sha=file_sha256(plan_file);waiting=time.time()
    failed=threading.Event();lock=threading.Lock();active={};cpu_active={};native_costs=[];cpu_costs=[]
    captures={};audits={};result={'status':'STOPPED_FUTURE_SPAN_QUALIFICATION'}
    cpu_pool=cf.ThreadPoolExecutor(2);deadline=None

    def frozen():
        if file_sha256(plan_file)!=plan_sha or any(file_sha256(p)!=sha for p,sha in plan['source_sha256'].items()):
            raise AssertionError('Frozen qualification plan/source changed')
        if file_sha256(args.root/plan['predecessor_plan'])!=plan['predecessor_plan_sha256']:
            raise AssertionError('Frozen predecessor plan changed')

    def guard():
        frozen()
        if failed.is_set() or (deadline is not None and time.time()>deadline):raise TimeoutError('Original qualification deadline/paired stop')

    def native(key,mode,half,parent=None):
        guard();identity=plan['identity']+'-'+key+'-'+mode;directory=args.root/'runs'/identity
        if (directory/'report.json').exists():
            report=json.loads((directory/'report.json').read_text())
            closure=args.root/'snapshots'/identity/'closure.json'
            if not closure.exists() or json.loads(closure.read_text())['exit_code']!=0:raise ValueError('Completed native receipt has no successful closure')
            for filename,key_sha in [('after121.pt','final_sha256'),('full-state-u120.pt','full_checkpoint_sha256')]:
                if file_sha256(directory/filename)!=report[key_sha]:raise AssertionError('Reused native checkpoint changed')
        else:
            if directory.exists():raise RuntimeError('Preserve partial native phase;explicit recovery identity required: '+identity)
            reclaim_host();container=create_compact_worker(args.root,identity,half);process=None;begin=time.time()
            with lock:active[identity]=container
            try:
                limit=min(deadline,begin+plan['native_phase_seconds_limit'])
                env=dict(os.environ,RESEARCH_ROOT=str(args.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                    RESEARCH_RUN=identity,RESEARCH_OWNER=identity,RESEARCH_CONTAINER=container,
                    RESEARCH_ENTRY_MODULE='scripts.qualify_future_span_native',RESEARCH_DDP_RANKS='32',
                    NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,half*32+32))),
                    RESEARCH_TIMEOUT_SECONDS=str(max(1,int(limit-time.time()))))
                options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/future-span-qualification-v1/plan.json',
                    '--plan-sha',plan_sha,'--case',key,'--mode',mode,'--comm-port',str(plan['root_comm_ports'][identity])]
                if parent:options+=['--resume','/work/instance-nvme/runs/'+parent[0]+'/full-state-u120.pt',
                                    '--resume-sha',parent[1]['full_checkpoint_sha256']]
                with (out/(identity+f'-{int(time.time())}.log')).open('x') as log:
                    process=subprocess.Popen([os.sys.executable,'-u','-m','scripts.supervise_isolated',*options],
                        env=env,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    guard()
                    if time.time()>limit:raise TimeoutError('Native qualification phase cap: '+identity)
                    time.sleep(5)
                if process.returncode:raise RuntimeError('Native qualification phase failed: '+identity)
                report=json.loads((directory/'report.json').read_text())
            finally:
                cancel_worker_container(container,identity,'qualification_phase_complete')
                if process and process.poll() is None:
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.terminate()
                closure=args.root/'snapshots'/identity/'closure.json'
                with lock:
                    active.pop(identity,None)
                    native_costs.append({'identity':identity,**json.loads(closure.read_text())} if closure.exists()
                        else {'identity':identity,'closure_missing':True,'wall_seconds':time.time()-begin})
        if (report['status']!='CAPTURED_FUTURE_SPAN_TRAINING_QUALIFICATION' or report['signature']['plan_sha256']!=plan_sha
                or report['case']!=plan['cases'][key] or report['mode']!=mode or not report['all_complete_replica_states_equal']):
            raise AssertionError('Completed native qualification identity/state differs')
        if parent and (not report['complete_state_on_load_exact'] or not report['cpu_rng_on_load_exact']
                or report['recovered_state_digest']!=parent[1]['full_checkpoint_state_digest']):
            raise AssertionError('Exact fresh-process recovery failed')
        return identity,report

    def cpu(key,mode,update,capture,reuse=None):
        guard();begin=time.time();target=out/'cpu'/key/f'{mode}-u{update}'
        process=None;reused=(target/'report.json').exists()
        try:
            if reused:report=json.loads((target/'report.json').read_text())
            else:
                if target.exists():raise RuntimeError('Preserve partial CPU audit;explicit recovery required')
                target.parent.mkdir(parents=True,exist_ok=True)
                options=['--root',str(args.root),'--plan',str(plan_file),'--plan-sha',plan_sha,
                    '--capture',str(args.root/'runs'/capture[0]),'--update',str(update),
                    '--output',str(target),'--upstream',str(args.root/'upstream-parcae-69284c1'),
                    '--seconds-limit',str(plan['cpu_phase_seconds_limit'])]
                if reuse:options+=['--reuse-gold',str(reuse)]
                env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12')
                with (out/(key+f'-{mode}-u{update}-{int(time.time())}.log')).open('x') as log:
                    process=subprocess.Popen([os.sys.executable,'-u','-m','scripts.audit_future_span_qualification',*options],
                        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with lock:cpu_active[str(target)]=process
                while process.poll() is None:
                    guard()
                    if time.time()>begin+plan['cpu_phase_seconds_limit']:raise TimeoutError('Independent CPU phase cap')
                    time.sleep(5)
                if process.returncode:raise RuntimeError('Independent CPU qualification failed: '+key)
                report=json.loads((target/'report.json').read_text())
            if (report['status']!='PASS_INDEPENDENT_FUTURE_SPAN_QUALIFICATION_AUDIT'
                    or report['gold_fingerprint']['plan_sha256']!=plan_sha or report['case']!=plan['cases'][key]
                    or report['native_mode']!=mode or report['update']!=update):
                raise AssertionError('Independent qualification receipt differs')
            return target,report
        except BaseException:failed.set();raise
        finally:
            if process and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            with lock:
                cpu_active.pop(str(target),None);cpu_costs.append({'identity':key+f'-{mode}-u{update}',
                    'assigned_chips':0,'wall_seconds':time.time()-begin,'reused_completed_receipt':reused,
                    'exit_code':process.returncode if process else None})

    tasks=queue.Queue()
    for key in plan['cases']:tasks.put(key)
    def worker(half):
        try:
            while not tasks.empty():
                try:key=tasks.get_nowait()
                except queue.Empty:return
                warm=native(key,'warm',half)
                initial=cpu_pool.submit(cpu,key,'warm',0,warm)
                trained=cpu_pool.submit(cpu,key,'warm',120,warm)
                replay=native(key,'replay',half,warm)
                with lock:
                    captures[key]={'warm':warm,'replay':replay};audits[key]={'initial':initial,'trained':trained}
                    write_state(public/'native-progress.json',{'records':[{k:v[1] for k,v in pair.items()} for pair in captures.values()]})
        except BaseException:failed.set();raise

    execution=None
    try:
        frozen()
        if started_file.exists():
            if not args.resume:raise FileExistsError('Existing qualification execution requires --resume')
            execution=json.loads(started_file.read_text())
            if execution['plan_sha256']!=plan_sha:raise AssertionError('Resume changed frozen execution plan')
            deadline=execution['deadline_unix'];guard()
        else:
            if args.resume:raise ValueError('No admitted execution exists to resume')
            while True:
                guard();predecessor_execution(args.root)
                folder=Path('results/parcae_low_depth_pilot_20261024/main-recovery-v2')
                ready=all((folder/f).exists() for f in ['training-progress.json','gradient-progress.json'])
                if ready:
                    ready=completed_native_receipts(args.root,json.loads((folder/'training-progress.json').read_text()),
                                                     json.loads((folder/'gradient-progress.json').read_text()))
                if ready and not managed_running():break
                if time.time()>plan['predecessor_deadline_unix']:
                    raise TimeoutError('Predecessor did not complete within its original deadline')
                write_state(public/'status.json',{'status':'WAITING_FOR_ALL_PREDECESSOR_NATIVE_RECEIPTS','assigned_chips':0})
                time.sleep(10)
            admitted=time.time();deadline=admitted+plan['execution_seconds_limit']
            execution={'plan_sha256':plan_sha,'admitted_unix':admitted,'deadline_unix':deadline,
                       'predecessor_wait_seconds':admitted-waiting}
            write_state(started_file,execution)
        write_state(public/'status.json',{'status':'RUNNING_FINITE_FUTURE_SPAN_QUALIFICATION','maximum_chips':16})
        with cf.ThreadPoolExecutor(2) as pool:
            for future in cf.as_completed([pool.submit(worker,half) for half in [0,1]]):future.result()
        records=[]
        for key in plan['cases']:
            initial=audits[key]['initial'].result(timeout=max(1,deadline-time.time()))
            trained=audits[key]['trained'].result(timeout=max(1,deadline-time.time()))
            replay=cpu_pool.submit(cpu,key,'replay',120,captures[key]['replay'],trained[0])
            records.append({'case':plan['cases'][key],'native':{k:v[1] for k,v in captures[key].items()},
                'initial_cpu':initial[1],'trained_cpu':trained[1],
                'replay_cpu':replay.result(timeout=max(1,deadline-time.time()))[1]})
            write_state(public/'progress.json',{'records':records})
        guard();result={'status':'PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION','records':records,
            'scientific_training_launched':False,'scope':plan['scope']}
    except BaseException as error:failed.set();result['error']=str(error);raise
    finally:
        failed.set()
        for identity,container in list(active.items()):
            try:cancel_worker_container(container,identity,'qualification_controller_stop')
            except BaseException:pass
        for process in list(cpu_active.values()):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        cpu_pool.shutdown(wait=True,cancel_futures=True)
        result.update(plan_sha256=plan_sha,qualification_only=True,research_optimizer_updates=0,test_accessed=False,
            execution=execution,wall_seconds_since_controller=time.time()-waiting)
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0})
        # Reconstruct all immutable phase closures,including work completed
        # before a controller-only restart. Reuse must never erase prior costs.
        all_native=[]
        for identity in plan['root_comm_ports']:
            folder=args.root/'snapshots'/identity;launch=folder/'launch.json';closure=folder/'closure.json'
            if not launch.exists():continue
            if closure.exists():all_native.append({'identity':identity,**json.loads(closure.read_text())})
            else:
                original=json.loads(launch.read_text());seconds=max(0,time.time()-original['started_unix'])
                all_native.append({'identity':identity,'closure_missing':True,'wall_seconds_through_collection':seconds,
                    'assigned_chip_hours_through_collection':seconds*len(original['physical_chips'])/3600})
        all_cpu=[]
        for file in sorted((out/'cpu').glob('*/*/report.json')):
            receipt=json.loads(file.read_text())
            all_cpu.append({'identity':file.parent.parent.name+'-'+file.parent.name,
                'status':receipt['status'],'assigned_chips':0,'wall_seconds':receipt['wall_seconds'],
                'prescribed_cpu_reference_optimizer_steps':receipt['prescribed_cpu_reference_optimizer_steps']})
        write_state(public/'execution-costs.json',{'native_attempts':all_native,'completed_cpu_audits_all_controller_attempts':all_cpu,
            'current_controller_native_phase_lifecycle':native_costs,'current_controller_cpu_jobs':cpu_costs,
            'wall_seconds_since_controller':time.time()-waiting,'execution':execution,
            'admitted_whole_instance_opportunity_chip_hours':max(0,time.time()-execution['admitted_unix'])*16/3600 if execution else 0,
            'scope':'Actual native phase closures and CPU jobs,including failures;predecessor waiting overlaps its reservation and is separate from admitted qualification time. Completed receipts reused on resume add no new native work. Whole-instance opportunity cost must not be added twice to overlapping campaigns.'})


if __name__=='__main__':main()
