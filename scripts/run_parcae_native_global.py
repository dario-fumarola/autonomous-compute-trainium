"""Wait for the frozen science to close, then qualify both staged gradients."""
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

from compute_trainium.execution_isolation import cancel_worker_container,managed_running
from compute_trainium.owned_native_phase import create_worker
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def compare(reference,candidate):
    expected=torch.load(reference,map_location='cpu',mmap=True,weights_only=False)
    actual=torch.load(candidate,map_location='cpu',mmap=True,weights_only=False)
    if set(expected['gradients'])!=set(actual['gradients']):
        raise AssertionError('Gradient parameter names differ')
    difference=norm=dot=actual_norm=maximum=0.
    contributors=[]
    for name,gradient in expected['gradients'].items():
        value=actual['gradients'][name]
        if gradient.shape!=value.shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise AssertionError('Native gradient shape/dtype/finiteness differs')
        left,right=gradient.double(),value.double()
        error=float((left-right).square().sum())
        square=float(left.square().sum())
        difference+=error;norm+=square
        dot+=float((left*right).sum());actual_norm+=float(right.square().sum())
        maximum=max(maximum,float((left-right).abs().max()))
        contributors.append({'parameter':name,'squared_error':error,'reference_squared_norm':square})
    return {'relative_l2':(difference/max(norm,1e-30))**.5,
        'cosine':dot/max((norm*actual_norm)**.5,1e-30),'maximum_absolute_error':maximum,
        'native_minus_cpu_nll':actual['loss']-expected['loss'],
        'largest_error_contributors':sorted(contributors,key=lambda r:-r['squared_error'])[:10]}


def launch_ready(terminal,managed_active,now,latest):
    if now>latest:
        raise TimeoutError('Qualification latest-start window expired')
    if terminal and terminal['status']!='COMPLETE_COMMON_FP32_DEPTH_CONNECTION_COMPARISON':
        raise RuntimeError('Scientific prerequisite stopped; no native qualification starts')
    return bool(terminal) and not managed_active


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    out=args.root/'parcae-native-global-v1'
    public=Path('results/parcae_native_global_20261024')
    cpu=json.loads((out/'report.json').read_text())
    if cpu['status']!='COMPLETE_INDEPENDENT_PARCAE_CPU_GLOBAL_REFERENCES':
        raise ValueError('Independent CPU references incomplete')
    reference_plan=json.loads((out/'plan.json').read_text())
    for path,sha in reference_plan['source_sha256'].items():
        if file_sha256(path)!=sha:
            raise AssertionError('CPU reference source changed')
    for name,path,status in [('stages','results/parcae_stages_cpu_20261024/attempt2/report.json','PASS_CPU_FULLGRAPH_STATIC_STAGES_AND_AUTOGRAD'),
                             ('optimizer','results/parcae_optimizer_cpu_20261024/attempt2-state-and-aot/report.json','PASS_CPU_RELEASE_OPTIMIZER_UPDATE_AND_RECOVERY_CHECKS')]:
        report=json.loads(Path(path).read_text())
        if report['status']!=status:
            raise AssertionError(name+' preparation gate failed')
        for file,sha in report['source_sha256'].items():
            if file_sha256(file)!=sha:raise AssertionError(name+' source changed')
    latest=dt.datetime(2026,10,25,1,tzinfo=dt.timezone.utc).timestamp()
    files=['scripts/run_parcae_native_global.py','scripts/capture_parcae_stages_global.py',
        'compute_trainium/execution_isolation.py','compute_trainium/owned_native_phase.py',
        'scripts/supervise_isolated.py','scripts/run_native_isolated.sh']
    plan_path=out/'native-execution-plan.json'
    if not plan_path.exists():
        plan={'identity':'parcae-native-global-v1-execution-r2','execution_revision':2,'registered_unix':time.time(),
            'source_sha256':{**reference_plan['source_sha256'],**{f:file_sha256(f) for f in files}},
            'reference_plan_sha256':file_sha256(out/'plan.json'),
            'cpu_report_sha256':file_sha256(out/'report.json'),
            'prerequisite_identity':'depth-connection-v1','latest_native_start_unix':latest,
            'native_window_seconds':1800,'local_gradient_gate':.02,'global_gradient_gate':.02,'nll_gate':.003,
            'optimizer_updates':0,'test_accessed':False,
            'scope':'Bounded paired native gradient qualification after the registered scientific campaign completes. No early chip reservation, optimizer, learning run or scientific success claim.'}
        write_state(plan_path,plan);write_state(public/'native-execution-plan.json',plan)
    plan=json.loads(plan_path.read_text())
    def frozen():
        if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):
            raise AssertionError('Qualification source changed')
        if file_sha256(out/'plan.json')!=plan['reference_plan_sha256'] or file_sha256(out/'report.json')!=plan['cpu_report_sha256']:
            raise AssertionError('Independent CPU proof changed')
    frozen()
    if args.prepare_only:return
    if (out/'native-report.json').exists():raise FileExistsError('No silent native rerun')
    waiting_started=time.time()
    execution_started=None
    active={};costs=[];lock=threading.Lock();records=[];result=None
    def native(arm,half,deadline):
        frozen()
        if time.time()>plan['latest_native_start_unix']:
            raise TimeoutError('Native launch decision is past the registered latest start')
        identity='parcae-native-global-v1-r2-'+arm
        if (args.root/'runs'/identity).exists():raise FileExistsError(identity)
        container=create_worker(args.root,identity,half)
        with lock:
            active[identity]=container
            write_state(public/'native-status.json',{'status':'RUNNING_NATIVE_GRADIENT_QUALIFICATION',
                'execution_revision':2,'observed_unix':time.time(),'assigned_chips':8*len(active),'scope':plan['scope']})
        proc=None
        try:
            env=dict(os.environ,RESEARCH_ROOT=str(args.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                RESEARCH_RUN=identity,RESEARCH_OWNER=identity,RESEARCH_CONTAINER=container,
                RESEARCH_ENTRY_MODULE='scripts.capture_parcae_stages_global',RESEARCH_DDP_RANKS='32',
                NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,half*32+32))),
                RESEARCH_TIMEOUT_SECONDS=str(max(1,int(deadline-time.time()))))
            with (out/(identity+'.log')).open('x') as log:
                if time.time()>plan['latest_native_start_unix']:
                    raise TimeoutError('Container setup crossed the latest native launch decision')
                proc=subprocess.Popen([os.sys.executable,'-u','-m','scripts.supervise_isolated',
                    '--reference-root','/work/instance-nvme/parcae-native-global-v1',
                    '--corpus','/work/instance-nvme/natural-append-6500m-v2/source.json',
                    '--packed-corpus','/work/instance-nvme/natural-append-6500m-v2/packed','--arm',arm],
                    env=env,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if time.time()>deadline:raise TimeoutError('Paired native30-minute deadline')
                time.sleep(5)
            if proc.returncode:raise RuntimeError('Native gradient phase failed: '+arm)
            report=json.loads((args.root/'runs'/identity/'report.json').read_text())
            if not report['weights_unchanged'] or not report['all_replica_gradients_equal'] or report['real_targets']!=reference_plan['real_targets']:
                raise AssertionError('Native identity/count/replica gate failed')
            proof=next(r for r in cpu['records'] if r['arm']==arm)
            ref=out/arm/'global-cpu-fp32.pt'
            if file_sha256(ref)!=proof['global_gradient_file_sha256']:
                raise AssertionError('Independent global reference file changed')
            global_delta=compare(ref,args.root/'runs'/identity/'global-native.pt')
            local_delta=compare(out/arm/'local-cpu-fp32.pt',args.root/'runs'/identity/'local-native.pt')
            if time.time()>deadline:
                raise TimeoutError('Qualification comparison exceeded the registered native window')
            passed=all(row['relative_l2']<=.02 and abs(row['native_minus_cpu_nll'])<=.003 for row in [local_delta,global_delta])
            row={'arm':arm,'local':local_delta,'global':global_delta,'passes_registered_gates':passed,
                 'weights_unchanged':True,'all_replica_gradients_equal':True,'optimizer_updates':0}
            with lock:
                records.append(row);write_state(public/'native-progress.json',{'records':records})
            if not passed:raise AssertionError('Native full-gradient numerical gate failed: '+arm)
            return row
        finally:
            cancel_worker_container(container,identity,'phase_complete')
            if proc is not None and proc.poll() is None:
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.terminate()
            closure=args.root/'snapshots'/identity/'closure.json'
            with lock:
                active.pop(identity,None)
                costs.append({'identity':identity,**json.loads(closure.read_text())} if closure.exists() else {'identity':identity,'closure_missing':True})
    try:
        while True:
            frozen()
            prerequisite=args.root/plan['prerequisite_identity']/'report.json'
            terminal=json.loads(prerequisite.read_text()) if prerequisite.exists() else None
            if launch_ready(terminal,bool(managed_running()) if terminal else True,
                            time.time(),plan['latest_native_start_unix']):break
            write_state(public/'native-status.json',{'status':'WAITING_FOR_REGISTERED_SCIENCE_TO_CLOSE',
                'execution_revision':2,'observed_unix':time.time(),'assigned_chips':0,'scope':plan['scope']})
            time.sleep(30)
        execution_started=time.time()
        write_state(public/'native-status.json',{'status':'RUNNING_NATIVE_GRADIENT_QUALIFICATION',
            'execution_revision':2,'started_unix':execution_started,'assigned_chips':0,'maximum_chips':16,'scope':plan['scope']})
        deadline=execution_started+plan['native_window_seconds']
        with cf.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(native,arm,i,deadline) for i,arm in enumerate(['ordinary','recurrent'])]
            for future in cf.as_completed(futures):future.result()
        frozen()
        result={'status':'PASS_PAIRED_NATIVE_PARCAE_STATIC_STAGE_GLOBAL_GRADIENTS','records':records,
                'optimizer_updates':0,'test_accessed':False,'scope':plan['scope']}
    except BaseException as error:
        result={'status':'STOPPED_PARCAE_NATIVE_GLOBAL_QUALIFICATION','error':str(error),
                'records':records,'optimizer_updates':0,'test_accessed':False,'scope':plan['scope']}
        raise
    finally:
        for identity,container in list(active.items()):
            try:cancel_worker_container(container,identity,'qualification_stop')
            except BaseException:pass
        now=time.time()
        result.update(waiting_seconds=(execution_started or now)-waiting_started,
            active_execution_seconds=now-execution_started if execution_started else 0.,execution_revision=2)
        write_state(out/'native-report.json',result);write_state(public/'native-report.json',result)
        write_state(public/'native-status.json',{'status':result['status'],'execution_revision':2,
            'observed_unix':now,'assigned_chips':0,'scope':plan['scope']})
        write_state(public/'native-execution-costs.json',{'status':result['status'],'attempts':costs,
            'waiting_seconds':result['waiting_seconds'],'active_execution_seconds':result['active_execution_seconds'],
            'scope':'Waiting assigns zero chips and overlaps the scientific campaign. Native closed receipts include compilation/capture/failures; CPU reference preparation has separate costs. Do not add overlapping wall intervals.'})


if __name__=='__main__':main()
