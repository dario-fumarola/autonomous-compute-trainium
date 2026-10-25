"""Finite operator + complete mature-model gradient check of a local workaround."""
import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_admission import completed_native_receipts
from compute_trainium.future_span_execution import Execution
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def register(root):
    parent_file=root/'parcae-low-depth-pilot-v1/main-plan-recovery-v2.json';parent=json.loads(parent_file.read_text())
    fixture_dir=root/'sdpa-math2-cpu-v1';manifest_file=fixture_dir/'manifest.json'
    fixtures=json.loads(manifest_file.read_text());report=json.loads((fixture_dir/'report.json').read_text())
    if report['status']!='COMPLETE_CPU_MATH2_SDPA_REPRO' or report['manifest_sha256']!=file_sha256(manifest_file):
        raise ValueError('Complete independently generated CPU fixtures required')
    for key in ['fixtures','gold']:
        if file_sha256(fixture_dir/fixtures[key+'_file'])!=fixtures[key+'_sha256']:raise AssertionError('Fixture/gold file changed')
    identity='sdpa-math2-native-v1';out=root/identity;public=Path('results/sdpa_math2_investigation_20261025/native-v1')
    points={}
    for arm in ['fixed1','poisson2']:
        phase='low-depth-main-r2-'+arm+'-finish';native=json.loads((root/'runs'/phase/'report.json').read_text())
        row=next(r for r in native['checkpoint_records'] if r['update']==native['update'])
        points[arm]={'case_id':native['case_id'],'checkpoint':f'runs/{phase}/checkpoint-u{native["update"]}.pt',
            'checkpoint_sha256':row['checkpoint_sha256']}
    files=['scripts/run_sdpa_math2_qualification.py','scripts/repro_sdpa_math2_cpu.py','scripts/repro_sdpa_math2_native.py',
        'scripts/audit_sdpa_math2_native.py','scripts/capture_math_sdpa_gradient.py','scripts/capture_low_depth_gradient.py',
        'scripts/audit_low_depth_gradient.py','compute_trainium/low_depth_gradient_probe.py',
        'compute_trainium/parcae_math_sdpa_candidate.py','compute_trainium/future_span_execution.py',
        'compute_trainium/future_span_admission.py','tests/test_parcae_math_sdpa_candidate.py']
    sources={**parent['source_sha256'],**{f:file_sha256(f) for f in files}}
    if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Current pilot/qualification source changed')
    plan={'identity':identity,'registered_unix':time.time(),'purpose':'zero_update_low_depth_gradient_validation',
        'execution_variant':'fp16_body_explicit_fp32_sdpa','parent_plan':str(parent_file.relative_to(root)),
        'parent_plan_sha256':file_sha256(parent_file),'points':points,'source_sha256':sources,
        'worker_source_sha256':{f:sha for f,sha in sources.items() if f.startswith(('scripts/','compute_trainium/','configs/','third_party/'))},
        'fixture_directory':fixture_dir.name,'fixture_manifest':fixtures,'fixture_manifest_sha256':file_sha256(manifest_file),
        'root_comm_ports':reserve_ports(root,[identity+'-operator']+[identity+'-'+key for key in points]),
        'predecessor_deadline_unix':parent['registered_unix']+parent['seconds_limit'],
        'seconds_limit':5400,'phase_seconds_limit':900,'cpu_phase_seconds_limit':1200,
        'admission':'All nine predecessor fits/gates complete,or predecessor terminal with no active managed native owner. No inter-phase gap stealing.',
        'scope':'Bounded numerical qualification only. Actual compiled operator fixtures on32 ranks and next complete512-window gradients at two immutable mature checkpoints. Explicit FP32 SDPA inputs/output retained internally,FP16 body/FP32 vocabulary head unchanged. Installed SDK/live pilot untouched. No optimizer or scientific launch.'}
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    return out,public,plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args();out=a.root/'sdpa-math2-native-v1'
    public=Path('results/sdpa_math2_investigation_20261025/native-v1');plan_file=out/'plan.json'
    if plan_file.exists():plan=json.loads(plan_file.read_text())
    else:out,public,plan=register(a.root)
    if a.prepare_only:return
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal qualification;explicit recovery required')
    result={'status':'STOPPED_SDPA_MATH2_QUALIFICATION'};execution=None;cpu_pool=cf.ThreadPoolExecutor(2)
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
            native=execution.native(name,'scripts.repro_sdpa_math2_native',[
                '--root','/work/instance-nvme','--plan','/work/instance-nvme/sdpa-math2-native-v1/plan.json',
                '--plan-sha',sha,'--comm-port',str(plan['root_comm_ports'][name])],
                'CAPTURED_COMPILED_SDPA_NUMERICAL_FIXTURES')
            target=out/'cpu-operator'
            return execution.cpu('operator-audit','scripts.audit_sdpa_math2_native',[
                '--root',str(a.root),'--plan',str(plan_file),'--plan-sha',sha,'--capture',str(a.root/'runs'/name),
                '--output',str(target)],target/'report.json','PASS_COMPILED_SDPA_WORKAROUND_FIXTURES')
        def model(key):
            name=plan['identity']+'-'+key
            native=execution.native(name,'scripts.capture_math_sdpa_gradient',[
                '--root','/work/instance-nvme','--plan','/work/instance-nvme/sdpa-math2-native-v1/plan.json',
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
            except BaseException:execution.failed.set();raise
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
        result.update(plan_sha256=file_sha256(plan_file),research_optimizer_updates=0,test_accessed=False,
            scope=plan['scope']);write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0})


if __name__=='__main__':main()
