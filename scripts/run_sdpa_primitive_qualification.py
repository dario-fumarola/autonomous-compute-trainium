"""Finite operator + complete mature-model gradient check of a local workaround."""
import argparse
import concurrent.futures as cf
import json
import time
import threading
from pathlib import Path

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_admission import completed_native_receipts
from compute_trainium.future_span_execution import Execution
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def register(root):
    predecessor=root/'sdpa-math2-native-v2/plan.json'
    original=json.loads(predecessor.read_text())
    terminal=json.loads((predecessor.parent/'report.json').read_text())
    if terminal['status']!='STOPPED_SDPA_MATH2_QUALIFICATION':
        raise ValueError('Preserved failed numerical qualification required')
    if any(file_sha256(f)!=sha for f,sha in original['source_sha256'].items()):
        raise AssertionError('Original qualification source changed')
    identity='sdpa-math2-native-v3';out=root/identity
    public=Path('results/sdpa_math2_investigation_20261025/native-v3')
    additions=['scripts/run_sdpa_primitive_qualification.py','scripts/repro_sdpa_math2_native_primitives.py',
        'scripts/audit_sdpa_math2_native_primitives.py','scripts/capture_primitive_sdpa_gradient.py',
        'compute_trainium/parcae_primitive_sdpa_candidate.py','tests/test_parcae_primitive_sdpa_candidate.py']
    sources={**original['source_sha256'],**{f:file_sha256(f) for f in additions}}
    from compute_trainium.native_snapshot_manifest import split_snapshot_sources
    worker,host_only=split_snapshot_sources(sources)
    plan={**original,'identity':identity,'registered_unix':time.time(),
        'source_sha256':sources,'worker_source_sha256':worker,'host_only_source_sha256':host_only,
        'recovery_of_plan_sha256':file_sha256(predecessor),
        'execution_variant':'fp16_body_fp32_attention_primitives',
        'scope':'Finite numerical qualification only. Exact original8 fixtures on32 ranks,now including explicit FP32 matmul/causal-mask/softmax/value-matmul differentiated by PyTorch autograd instead of the installed SDPA custom backward. Same2% gradient/0.2% output gates. Complete512-window gradients at the same two mature checkpoints if fixtures pass. No optimizer updates,science,SDK changes or performance claim.',
        'root_comm_ports':reserve_ports(root,[identity+'-operator']+[identity+'-'+key for key in original['points']]),
        'execution_repair':'Prior cast-only workaround failed the compiled saturated fixture and is preserved. This independently registered alternative bypasses SDPA custom backward with explicit differentiable primitives. Original and cast-only paths retained as numerical controls. Same original fixtures/checkpoints/thresholds,no scientific outcomes.'}
    for key in ['fixtures','gold']:
        directory=root/plan['fixture_directory'];info=plan['fixture_manifest']
        if file_sha256(directory/info[key+'_file'])!=info[key+'_sha256']:
            raise AssertionError('Frozen numerical arrays changed')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    return out,public,plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args();out=a.root/'sdpa-math2-native-v3'
    public=Path('results/sdpa_math2_investigation_20261025/native-v3');plan_file=out/'plan.json'
    if plan_file.exists():plan=json.loads(plan_file.read_text())
    else:out,public,plan=register(a.root)
    if a.prepare_only:return
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal qualification;explicit recovery required')
    result={'status':'STOPPED_SDPA_MATH2_QUALIFICATION'};execution=None;cpu_pool=cf.ThreadPoolExecutor(2)
    task_failures=[];failure_lock=threading.Lock()
    try:
        while not (out/'execution/execution-start.json').exists():
            if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Queued numerical source changed')
            if file_sha256(a.root/plan['parent_plan'])!=plan['parent_plan_sha256']:raise AssertionError('Frozen predecessor changed')
            base=Path('results/parcae_low_depth_pilot_20261024/main-recovery-v2')
            ready=completed_native_receipts(a.root,json.loads((base/'training-progress.json').read_text()),
                json.loads((base/'gradient-progress.json').read_text()))
            parent_terminal=a.root/'parcae-low-depth-pilot-v1/main-execution-recovery-v2/report.json'
            if (ready or parent_terminal.exists()) and not managed_running():break
            if time.time()>plan['predecessor_deadline_unix']:raise TimeoutError('Original predecessor window expired')
            write_state(public/'status.json',{'status':'WAITING_FOR_PREDECESSOR_NATIVE_COMPLETION','assigned_chips':0});time.sleep(10)
        execution=Execution(a.root,plan_file,out/'execution',resume=a.resume);sha=execution.sha
        def operator():
            name=plan['identity']+'-operator'
            native=execution.native(name,'scripts.repro_sdpa_math2_native_primitives',[
                '--root','/work/instance-nvme','--plan','/work/instance-nvme/sdpa-math2-native-v3/plan.json',
                '--plan-sha',sha,'--comm-port',str(plan['root_comm_ports'][name])],
                'CAPTURED_COMPILED_SDPA_NUMERICAL_FIXTURES')
            target=out/'cpu-operator'
            return execution.cpu('operator-audit','scripts.audit_sdpa_math2_native_primitives',[
                '--root',str(a.root),'--plan',str(plan_file),'--plan-sha',sha,'--capture',str(a.root/'runs'/name),
                '--output',str(target)],target/'report.json','PASS_COMPILED_SDPA_WORKAROUND_FIXTURES')
        def model(key):
            name=plan['identity']+'-'+key
            native=execution.native(name,'scripts.capture_primitive_sdpa_gradient',[
                '--root','/work/instance-nvme','--plan','/work/instance-nvme/sdpa-math2-native-v3/plan.json',
                '--plan-sha',sha,'--point',key,'--comm-port',str(plan['root_comm_ports'][name])],
                'CAPTURED_ZERO_UPDATE_LOW_DEPTH_GLOBAL_GRADIENT')
            target=out/'cpu'/key
            options=['--root',str(a.root),'--plan',str(plan_file),'--plan-sha',sha,'--point',key,
                '--capture',str(a.root/'runs'/name),'--output',str(target),
                '--upstream',str(a.root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
            return native,cpu_pool.submit(execution.cpu,key+'-audit','scripts.audit_low_depth_gradient',options,
                target/'report.json','PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT')
        def guarded(function,*args):
            try:return function(*args)
            except BaseException as error:
                with failure_lock:task_failures.append({'task':function.__name__,'arguments':list(args),
                    'error':str(error),'unix':time.time()})
                execution.failed.set();raise
        with cf.ThreadPoolExecutor(2) as pool:
            op=pool.submit(guarded,operator)
            models={key:pool.submit(guarded,model,key) for key in ['poisson2','fixed1']}
            records=[]
            for key,future in models.items():
                native,audit=future.result(timeout=max(1,execution.deadline-time.time()))
                records.append({'point':key,'native':native,'independent_cpu':audit.result(timeout=max(1,execution.deadline-time.time()))})
                write_state(public/'gradient-progress.json',{'records':records})
            checked=op.result(timeout=max(1,execution.deadline-time.time()))
        execution.check();result={'status':'PASS_SDPA_OPERATOR_AND_MATURE_MODEL_GRADIENTS',
            'operator_audit':checked,'model_gradient_audits':records,'scientific_training_launched':False}
    except BaseException as error:
        if execution:execution.failed.set()
        result['error']=str(error);raise
    finally:
        if execution:execution.failed.set()
        cpu_pool.shutdown(wait=True,cancel_futures=True)
        if execution:execution.close(public/'execution-costs.json')
        result.update(task_failures=task_failures,primary_error=task_failures[0] if task_failures else None,
            plan_sha256=file_sha256(plan_file),research_optimizer_updates=0,test_accessed=False,
            scope=plan['scope']);write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0})


if __name__=='__main__':main()
