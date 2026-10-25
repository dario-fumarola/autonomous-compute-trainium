"""Bounded two-case numerical gate;never admits scientific training."""
import argparse
import concurrent.futures as cf
import copy
import json
import signal
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='small-core-mixing-qualification-v1'
PUBLIC=Path('results/small_core_mixing_qualification_20261025')


def register(root):
    old_file=root/'future-span-math-qualification-v2/plan.json'
    old=json.loads(old_file.read_text())
    prior=root/'future-span-math-qualification-v2/report.json'
    proof=root/'small-core-mixing-reference-cpu-v1/report.json'
    work=root/'future-span-work-control-v1/report.json'
    prefill=root/'future-span-prefill-native-v1/report.json'
    if (json.loads(prior.read_text())['status']!='PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION'
            or json.loads(proof.read_text())['status']!='PASS_INDEPENDENT_CPU_MIXING_CONTROL_FIXTURES'
            or json.loads(work.read_text())['status']!='COMPLETE_WITHIN_FAMILY_WORK_MATCHED_CONTROLS'
            or json.loads(prefill.read_text())['status']!='COMPLETE_READ_ONLY_SUPPORTED_DEPTH_PREFILL_MEASUREMENT'
            or managed_running()):
        raise RuntimeError('All original studies,CPU comparison and idle ownership required')
    additions=['scripts/run_mixing_control_qualification.py','scripts/qualify_mixing_control_native.py',
        'scripts/audit_mixing_control_cpu.py','scripts/screen_mixing_reference_cpu.py',
        'compute_trainium/parcae_mixing_control.py','compute_trainium/mixing_control_reference.py',
        'tests/test_parcae_mixing_control.py','docs/SMALL_CORE_MIXING_NATIVE_QUALIFICATION_20261025.md']
    sources={**old['source_sha256'],**{p:file_sha256(p) for p in additions}}
    if any(file_sha256(p)!=h for p,h in sources.items()):
        raise AssertionError('Original or new qualification source changed')
    worker,host=split_snapshot_sources(sources)
    cases={k:old['cases'][k] for k in ['one-ntp','recurrent-ntp']}
    names=[IDENTITY+'-'+k+'-'+mode for k in cases for mode in ['warm','replay']]
    plan=copy.deepcopy(old)
    plan.update(identity=IDENTITY,registered_unix=time.time(),cases=cases,
        model_variant='all_mixes_then_one_core_fullgrad',source_sha256=sources,
        worker_source_sha256=worker,host_only_source_sha256=host,
        seconds_limit=1800,phase_seconds_limit=600,cpu_phase_seconds_limit=900,
        root_comm_ports=reserve_ports(root,names),
        prerequisite_sha256={str(p.relative_to(root)):file_sha256(p) for p in [prior,proof,work,prefill]},
        scope=Path('docs/SMALL_CORE_MIXING_NATIVE_QUALIFICATION_20261025.md').read_text())
    out=root/IDENTITY;out.mkdir(exist_ok=False);PUBLIC.mkdir(parents=True,exist_ok=False)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan


def run(root,plan,*,resume=False):
    out=root/IDENTITY;plan_file=out/'plan.json'
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal qualification')
    if (any(file_sha256(p)!=h for p,h in plan['source_sha256'].items())
            or any(file_sha256(root/p)!=h for p,h in plan['prerequisite_sha256'].items())):
        raise AssertionError('Frozen original prerequisites or source changed')
    execution=Execution(root,plan_file,out/'execution',resume=resume)
    result=dict(status='STOPPED_CHEAP_MIXING_CONTROL_QUALIFICATION')
    pool=cf.ThreadPoolExecutor(4);records=[]
    def interrupted(signum,frame):
        execution.failed.set();raise KeyboardInterrupt('Mixing qualification signal '+str(signum))
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}

    def native(key,mode,parent=None):
        name=IDENTITY+'-'+key+'-'+mode
        options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
            '--plan-sha',execution.sha,'--case',key,'--mode',mode,
            '--comm-port',str(plan['root_comm_ports'][name])]
        if parent:options += ['--resume','/work/instance-nvme/runs/'+parent[0]+'/full-state-u120.pt',
                              '--resume-sha',parent[1]['full_checkpoint_sha256']]
        report=execution.native(name,'scripts.qualify_mixing_control_native',options,
            'CAPTURED_FUTURE_SPAN_TRAINING_QUALIFICATION')
        if (report['signature']['plan_sha256']!=execution.sha or report['case']!=plan['cases'][key]
                or not report['all_complete_replica_states_equal']):
            raise AssertionError('Owned numerical qualification identity/replicas differ')
        if parent and (not report['complete_state_on_load_exact'] or not report['cpu_rng_on_load_exact']
                or report['recovered_state_digest']!=parent[1]['full_checkpoint_state_digest']):
            raise AssertionError('Exact full model/optimizer/RNG recovery failed')
        return name,report

    def cpu(key,mode,update,capture,reuse=None):
        target=out/'cpu'/key/f'{mode}-u{update}'
        options=['--root',str(root),'--plan',str(plan_file),'--plan-sha',execution.sha,
            '--capture',str(root/'runs'/capture[0]),'--update',str(update),'--output',str(target),
            '--upstream',str(root/'upstream-parcae-69284c1'),'--seconds-limit','900']
        if reuse:options += ['--reuse-gold',str(reuse)]
        row=execution.cpu(key+f'-{mode}-u{update}','scripts.audit_mixing_control_cpu',options,
            target/'report.json','PASS_INDEPENDENT_FUTURE_SPAN_QUALIFICATION_AUDIT')
        if (row['gold_fingerprint']['plan_sha256']!=execution.sha or row['case']!=plan['cases'][key]
                or row['native_mode']!=mode or row['update']!=update):
            raise AssertionError('Independent control audit identity differs')
        return target,row

    def case(key):
        try:
            warm=native(key,'warm')
            initial=pool.submit(cpu,key,'warm',0,warm)
            trained=pool.submit(cpu,key,'warm',120,warm)
            replay=native(key,'replay',warm)
            return key,warm,replay,initial,trained
        except BaseException:
            execution.failed.set();raise
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_NUMERICAL_QUALIFICATION_ONLY',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as native_pool:
            captures=[f.result() for f in [native_pool.submit(case,k) for k in plan['cases']]]
        for key,warm,replay,initial,trained in captures:
            first=initial.result(timeout=max(1,execution.deadline-time.time()))
            mature=trained.result(timeout=max(1,execution.deadline-time.time()))
            replay_gold=cpu(key,'replay',120,replay,mature[0])
            records.append(dict(case=plan['cases'][key],model_variant=plan['model_variant'],
                native={'warm':warm[1],'replay':replay[1]},initial_cpu=first[1],
                trained_cpu=mature[1],replay_cpu=replay_gold[1]))
            write_state(PUBLIC/'progress.json',dict(records=records))
        execution.check()
        result=dict(status='PASS_COMPLETE_CHEAP_MIXING_CONTROL_QUALIFICATION',records=records,
            scientific_training_admitted=False,qualification_native_optimizer_updates=244)
    except BaseException as error:
        execution.failed.set();result.update(error=str(error),records=records);raise
    finally:
        execution.failed.set();pool.shutdown(wait=True,cancel_futures=True)
        execution.close(PUBLIC/'execution-costs.json')
        for s,h in handlers.items():signal.signal(s,h)
        result.update(plan_sha256=execution.sha,qualification_only=True,research_optimizer_updates=0,
            test_accessed=False,terminal_alpha_spent=0,terminal_success=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();file=args.root/IDENTITY/'plan.json'
    plan=json.loads(file.read_text()) if file.exists() else register(args.root)
    if not args.prepare_only:run(args.root,plan,resume=args.resume)


if __name__=='__main__':main()
