"""Bounded four-cell updates/recovery gate after the compiled attention check."""
import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


PUBLIC=Path('results/future_span_qualification_20261025/math-sdpa-v2')
IDENTITY='future-span-math-qualification-v2'


def register(root):
    previous=root/'future-span-qualification-v1/plan.json'
    old=json.loads(previous.read_text());terminal=json.loads((previous.parent/'report.json').read_text())
    if terminal['status']!='STOPPED_FUTURE_SPAN_QUALIFICATION' or (previous.parent/'started.json').exists():
        raise ValueError('Only the stopped-before-admission preparation is carried forward')
    if any(file_sha256(p)!=sha for p,sha in old['source_sha256'].items()):
        raise AssertionError('Unexecuted original preparation source changed')
    prerequisite=root/'sdpa-math2-native-v3/plan.json'
    sdpa=json.loads(prerequisite.read_text())
    additions=['scripts/run_future_span_math_qualification.py','scripts/qualify_future_span_math_sdpa.py',
        'compute_trainium/future_span_math_sdpa.py','tests/test_future_span_math_sdpa.py',
        'compute_trainium/future_span_execution.py','compute_trainium/native_snapshot_manifest.py']
    sources={**old['source_sha256'],**sdpa['source_sha256'],**{p:file_sha256(p) for p in additions}}
    worker,host=split_snapshot_sources(sources)
    names=[IDENTITY+'-'+key+'-'+mode for key in old['cases'] for mode in ['warm','replay']]
    plan={**old,'identity':IDENTITY,'registered_unix':time.time(),
        'execution_variant':'fp16_body_fp32_attention_primitives','source_sha256':sources,
        'worker_source_sha256':worker,'host_only_source_sha256':host,
        'replaces_unexecuted_plan_sha256':file_sha256(previous),
        'sdpa_prerequisite_plan':str(prerequisite.relative_to(root)),
        'sdpa_prerequisite_plan_sha256':file_sha256(prerequisite),
        'seconds_limit':5400,'phase_seconds_limit':900,'cpu_phase_seconds_limit':1200,
        'root_comm_ports':reserve_ports(root,names),
        'scope':'Numerical/throughput qualification only,after complete compiled operator and mature full-global-gradient pass. Same frozen four objective cells,qualification-only initial weights,natural TRAIN stream,121 warm updates plus one replay update per cell,complete512-window gradients at updates0/120,independent optimizer checks,exact saved-state/RNG recovery,100 warm varying-schedule steps. Explicit local differentiable FP32 attention primitives and unchanged FP16 body/FP32 heads/masters/moments. No scientific fit,held-out selection or automatic continuation.'}
    out=root/IDENTITY;out.mkdir(exist_ok=False);PUBLIC.mkdir(parents=True,exist_ok=False)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return out,plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.root/IDENTITY;plan_file=out/'plan.json'
    if plan_file.exists():plan=json.loads(plan_file.read_text())
    else:out,plan=register(args.root)
    if args.prepare_only:return
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal qualification;explicit recovery required')
    execution=None;result={'status':'STOPPED_FUTURE_SPAN_TRAINING_QUALIFICATION'}
    cpu_pool=cf.ThreadPoolExecutor(2);audits={};captures={};prerequisite_sha=None
    try:
        prerequisite=args.root/plan['sdpa_prerequisite_plan']
        report_file=prerequisite.parent/'report.json'
        while True:
            if (file_sha256(prerequisite)!=plan['sdpa_prerequisite_plan_sha256']
                    or any(file_sha256(p)!=sha for p,sha in plan['source_sha256'].items())):
                raise AssertionError('Frozen attention/qualification sources changed')
            if report_file.exists():
                prior=json.loads(report_file.read_text())
                if (prior['status']!='PASS_SDPA_OPERATOR_AND_MATURE_MODEL_GRADIENTS'
                        or prior['plan_sha256']!=plan['sdpa_prerequisite_plan_sha256']):
                    raise RuntimeError('Attention qualification did not pass;future objective stays untrained')
                if not managed_running():prerequisite_sha=file_sha256(report_file);break
            if time.time()>plan['registered_unix']+5400:raise TimeoutError('Finite attention-prerequisite window expired')
            write_state(PUBLIC/'status.json',{'status':'WAITING_FOR_COMPLETE_COMPILED_ATTENTION_CHECK','assigned_chips':0})
            time.sleep(10)
        execution=Execution(args.root,plan_file,out/'execution',resume=args.resume);sha=execution.sha
        write_state(PUBLIC/'status.json',{'status':'RUNNING_NUMERICAL_UPDATES_AND_RECOVERY','maximum_chips':16})

        def native(key,mode,parent=None):
            name=IDENTITY+'-'+key+'-'+mode
            options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
                '--plan-sha',sha,'--case',key,'--mode',mode,'--comm-port',str(plan['root_comm_ports'][name])]
            if parent:options+=['--resume','/work/instance-nvme/runs/'+parent[0]+'/full-state-u120.pt',
                                '--resume-sha',parent[1]['full_checkpoint_sha256']]
            report=execution.native(name,'scripts.qualify_future_span_math_sdpa',options,
                'CAPTURED_FUTURE_SPAN_TRAINING_QUALIFICATION')
            if (report['signature']['plan_sha256']!=sha or report['case']!=plan['cases'][key]
                    or not report['all_complete_replica_states_equal']):raise AssertionError('Qualification identity/replicas differ')
            if parent and (not report['complete_state_on_load_exact'] or not report['cpu_rng_on_load_exact']
                    or report['recovered_state_digest']!=parent[1]['full_checkpoint_state_digest']):
                raise AssertionError('Exact full-state/RNG recovery failed')
            return name,report

        def cpu(key,mode,update,capture,reuse=None):
            target=out/'cpu'/key/f'{mode}-u{update}'
            options=['--root',str(args.root),'--plan',str(plan_file),'--plan-sha',sha,
                '--capture',str(args.root/'runs'/capture[0]),'--update',str(update),'--output',str(target),
                '--upstream',str(args.root/'upstream-parcae-69284c1'),'--seconds-limit','1200']
            if reuse:options+=['--reuse-gold',str(reuse)]
            report=execution.cpu(key+f'-{mode}-u{update}','scripts.audit_future_span_qualification',options,
                target/'report.json','PASS_INDEPENDENT_FUTURE_SPAN_QUALIFICATION_AUDIT')
            if (report['gold_fingerprint']['plan_sha256']!=sha or report['case']!=plan['cases'][key]
                    or report['native_mode']!=mode or report['update']!=update):raise AssertionError('Independent audit identity differs')
            return target,report

        def case(key):
            try:
                warm=native(key,'warm')
                initial=cpu_pool.submit(cpu,key,'warm',0,warm)
                trained=cpu_pool.submit(cpu,key,'warm',120,warm)
                replay=native(key,'replay',warm)
                captures[key]={'warm':warm,'replay':replay};audits[key]={'initial':initial,'trained':trained}
                write_state(PUBLIC/(key+'-native.json'),{k:v[1] for k,v in captures[key].items()})
            except BaseException:execution.failed.set();raise

        with cf.ThreadPoolExecutor(2) as pool:
            for future in cf.as_completed([pool.submit(case,key) for key in plan['cases']]):future.result()
        records=[]
        for key in plan['cases']:
            initial=audits[key]['initial'].result(timeout=max(1,execution.deadline-time.time()))
            trained=audits[key]['trained'].result(timeout=max(1,execution.deadline-time.time()))
            replay=cpu_pool.submit(cpu,key,'replay',120,captures[key]['replay'],trained[0])
            records.append({'case':plan['cases'][key],'native':{k:v[1] for k,v in captures[key].items()},
                'initial_cpu':initial[1],'trained_cpu':trained[1],
                'replay_cpu':replay.result(timeout=max(1,execution.deadline-time.time()))[1]})
            write_state(PUBLIC/'progress.json',{'records':records})
        execution.check();result={'status':'PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION',
            'records':records,'scientific_training_launched':False}
    except BaseException as error:
        if execution:execution.failed.set()
        result['error']=str(error);raise
    finally:
        if execution:execution.failed.set()
        cpu_pool.shutdown(wait=True,cancel_futures=True)
        if execution:execution.close(PUBLIC/'execution-costs.json')
        result.update(plan_sha256=file_sha256(plan_file),prerequisite_receipt_sha256=prerequisite_sha,
            qualification_only=True,research_optimizer_updates=0,test_accessed=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',{'status':result['status'],'assigned_chips':0})


if __name__=='__main__':main()
