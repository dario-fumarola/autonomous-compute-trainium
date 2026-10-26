"""A fixed-two mixing-only control; reuse the qualified finite tuning engine."""
import argparse
import copy
import json
from pathlib import Path
import signal
import time

import numpy as np

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.mixing_control_science import matrix_work, VARIANT
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_fixed_two_ntp_tune as engine

IDENTITY='fixed-two-mixing-tune-v1'
PREPARATION='fixed-two-mixing-tune-preparation-v1'
QUEUE='fixed-two-mixing-tune-queue-v1'
PUBLIC=Path('results/fixed_two_mixing_control_20261026/tune')
DOC='docs/FIXED_TWO_MIXING_CONTROL_20261026.md'


class MixingExecution(engine.GuardedExecution):
    def native(self,identity,module,options,status):
        mapped={'scripts.train_future_span_primitives':'scripts.train_mixing_control_scientific_v2',
                'scripts.capture_future_span_primitives':'scripts.capture_mixing_control_gradient'}
        return super().native(identity,mapped.get(module,module),options,status)

    def cpu(self,identity,module,options,report_file,status):
        mapped={'scripts.evaluate_future_span_cpu':'scripts.evaluate_mixing_prediction_v2',
                'scripts.audit_future_span_gradient':'scripts.audit_mixing_scientific_gradient'}
        return super().cpu(identity,mapped.get(module,module),options,report_file,status)


def prepare(root):
    out=root/PREPARATION;file=out/'manifest.json'
    if file.exists():
        existing=json.loads(file.read_text())
        if any(file_sha256(p)!=h for p,h in existing['source_sha256'].items()):
            raise AssertionError('Preserve the original prepared control sources')
        return existing
    started=time.time();source_file=root/'fixed-two-ntp-tune-v1/plan.json';source=json.loads(source_file.read_text())
    mixing_file=root/'small-core-mixing-tune-v2/plan.json';mixing=json.loads(mixing_file.read_text())
    hardware_file=root/'small-core-mixing-qualification-v1/report.json';hardware=json.loads(hardware_file.read_text())
    if hardware['status']!='PASS_COMPLETE_CHEAP_MIXING_CONTROL_QUALIFICATION':
        raise ValueError('Completed exact-operator qualification required')
    additions=['scripts/run_fixed_two_mixing_tune.py',DOC,'scripts/evaluate_mixing_prediction_v2.py',
               'compute_trainium/mixing_prediction_reference_v2.py']
    sources={**source['source_sha256']}
    for path,sha in {**mixing['source_sha256'],**{p:file_sha256(p) for p in additions}}.items():
        if path in sources and sources[path]!=sha:raise AssertionError('Original frozen source versions conflict')
        sources[path]=sha
    if any(file_sha256(p)!=h for p,h in sources.items()):raise AssertionError('Qualified scientific stack changed')
    for initial in source['initial_states'].values():
        if file_sha256(source_file.parent/initial['file'])!=initial['sha256']:
            raise AssertionError('Original unfitted initialization changed')
    measured=next(r['native']['warm'] for r in hardware['records'] if r['case']['arm']=='recurrent')
    if measured['warm_updates']!=100:raise ValueError('Measured100 warm complete updates required')
    step=measured['warm_seconds']/100;cold=max(120.,1.5*(measured['wall_seconds']-measured['warm_seconds']))
    predicted=1728*step*1.5/2+10*cold/2+4*900/3+3*360/3+300
    if predicted>5400:raise RuntimeError('Complete fixed three-rate tuning does not fit')
    c=ParcaeCandidateConfig(**source['model']);span=FutureSpanConfig(**source['future_config'])
    example=next(iter(source['cases'].values()));depth_file=source_file.parent/example['depth_table']
    if file_sha256(depth_file)!=example['depth_table_sha256']:raise AssertionError('Provisioned depth table changed')
    depths=np.full_like(np.load(depth_file,mmap_mode='r'),2)
    original_manifest=root/'future-span-study-preparation-v1/manifest.json';original=json.loads(original_manifest.read_text())
    work=matrix_work(c,span,depths,world=32,optimizer_per_replica=original['optimizer_matrix_ops_per_replica'])
    out.mkdir();np.save(out/'fixed-two-mixes.npy',depths);np.save(out/'fixed-two-mixing-work.npy',work)
    if not np.all(np.load(out/'fixed-two-mixes.npy')==2) or not np.array_equal(np.load(out/'fixed-two-mixing-work.npy'),work):
        raise AssertionError('Forced mixing/work table readback differs')
    manifest=dict(status='PASS_FIXED_TWO_MIXING_CPU_PREPARATION_ONLY',identity=PREPARATION,
        source_plan=str(source_file.relative_to(root)),source_plan_sha256=file_sha256(source_file),
        source_sha256=sources,hardware_report_sha256=file_sha256(hardware_file),
        original_manifest_sha256=file_sha256(original_manifest),depth_file='fixed-two-mixes.npy',
        depth_sha256=file_sha256(out/'fixed-two-mixes.npy'),work_file='fixed-two-mixing-work.npy',
        work_sha256=file_sha256(out/'fixed-two-mixing-work.npy'),work_per_update=int(work[0]),
        cases=list(source['cases']),rates=sorted(c['peak_lr'] for c in source['cases'].values()),
        measured_budget=dict(reference_warm_seconds_per_update=step,cold_allowance=cold,
            initial_seconds_with_margin=predicted),optimizer_updates=0,model_forwards=0,assigned_chips=0,
        native_launch_allowed=False,test_accessed=False,terminal_alpha_spent=0,wall_seconds=time.time()-started,
        scope=Path(DOC).read_text())
    if len(source['cases'])!=3 or set(manifest['rates'])!={.008,.016,.032}:raise ValueError('Fixed fair selection grid required')
    PUBLIC.mkdir(parents=True);write_state(file,manifest);write_state(PUBLIC/'preparation.json',manifest)
    return manifest


def queue_plan(root,manifest):
    out=root/QUEUE;file=out/'plan.json'
    if file.exists():return json.loads(file.read_text())
    main_clock=json.loads((root/'fixed-two-ntp-main-v1/execution/execution-start.json').read_text())
    report=root/'fixed-two-ntp-main-v1/first-primary-pair.json';first=json.loads(report.read_text())
    if first['status']!='AUDITED_FIRST_PRIMARY_PAIR_REMAINING_CURVE_RUNNING':raise ValueError('Actual audited primary discovery required')
    queue=dict(identity=QUEUE,registered_unix=time.time(),source_sha256=manifest['source_sha256'],
        preparation_manifest_sha256=file_sha256(root/PREPARATION/'manifest.json'),
        observed_primary_pair_sha256=file_sha256(report),main_plan_sha256=file_sha256(root/'fixed-two-ntp-main-v1/plan.json'),
        wait_deadline_unix=main_clock['deadline_unix']+5*60+1800,assigned_chips_while_waiting=0,
        optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0,
        scope='Wait without chips for the complete fixed-core curve and its complete prefill measurement. Admit only the fixed three-rate mixing-only TUNE under its own measured90-minute cap. No main fit,confirmation or quality-based grid expansion follows automatically.')
    out.mkdir();write_state(file,queue);write_state(PUBLIC/'queue-plan.json',queue)
    return queue


def register(root,manifest,queue):
    if managed_running():raise RuntimeError('Complete predecessor native ownership required')
    source_file=root/manifest['source_plan'];source=json.loads(source_file.read_text())
    for path,sha in [(source_file,manifest['source_plan_sha256']),
                     (root/PREPARATION/'fixed-two-mixes.npy',manifest['depth_sha256']),
                     (root/PREPARATION/'fixed-two-mixing-work.npy',manifest['work_sha256'])]:
        if file_sha256(path)!=sha:raise AssertionError('Prepared input changed')
    out=root/IDENTITY;out.mkdir();plan=copy.deepcopy(source);work=np.load(root/PREPARATION/'fixed-two-mixing-work.npy',mmap_mode='r')
    for case in plan['cases'].values():
        case.update(depth_table='../'+PREPARATION+'/fixed-two-mixes.npy',depth_table_sha256=manifest['depth_sha256'],
            work_table='../'+PREPARATION+'/fixed-two-mixing-work.npy',work_table_sha256=manifest['work_sha256'],
            training_dense_matrix_ops=int(work[:576].sum()))
    worker,host=split_snapshot_sources(manifest['source_sha256'])
    names=[IDENTITY+'-'+key+'-'+phase for key in plan['cases'] for phase in ['prefix','prefix-gradient','finish','final-gradient']]
    plan.update(identity=IDENTITY,registered_unix=time.time(),model_variant=VARIANT,
        mixing_depth_policy='exactly_two_full_gradient_mixes_then_one_core_execution',
        source_sha256=manifest['source_sha256'],worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),seconds_limit=5400,phase_seconds_limit=1200,cpu_phase_seconds_limit=1200,
        measured_budget=manifest['measured_budget'],scope=manifest['scope'],
        preparation_manifest_sha256=queue['preparation_manifest_sha256'],
        cheap_qualification_report_sha256=manifest['hardware_report_sha256'],
        completed_fixed_core_main_sha256=file_sha256(root/'fixed-two-ntp-main-v1/report.json'),
        completed_prefill_sha256=file_sha256(root/'fixed-two-prefill-native-v1/report.json'))
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    def interrupted(number,frame):raise KeyboardInterrupt('Fixed mixing control queue signal '+str(number))
    for number in [signal.SIGINT,signal.SIGTERM]:signal.signal(number,interrupted)
    manifest=prepare(a.root);queue=queue_plan(a.root,manifest)
    if a.prepare_only:return
    try:
        if not a.resume:
            if (a.root/QUEUE/'report.json').exists():raise FileExistsError('Preserve terminal queue receipt')
            while True:
                if time.time()>queue['wait_deadline_unix']:raise TimeoutError('Original control admission window expired')
                if (file_sha256(a.root/PREPARATION/'manifest.json')!=queue['preparation_manifest_sha256']
                        or file_sha256(a.root/'fixed-two-ntp-main-v1/first-primary-pair.json')!=queue['observed_primary_pair_sha256']
                        or file_sha256(a.root/'fixed-two-ntp-main-v1/plan.json')!=queue['main_plan_sha256']
                        or any(file_sha256(f)!=h for f,h in queue['source_sha256'].items())):
                    raise AssertionError('Frozen control inputs changed')
                required=[(a.root/'fixed-two-ntp-main-v1/report.json','COMPLETE_FIXED_TWO_NTP_CURVE'),
                          (a.root/'fixed-two-prefill-native-v1/report.json','COMPLETE_FIXED_CURVE_PREFILL_MEASUREMENT')]
                complete=True
                for file,status in required:
                    if file.exists():
                        if json.loads(file.read_text())['status']!=status:raise RuntimeError('Predecessor stopped;no automatic control admission')
                    else:complete=False
                if complete and not managed_running():break
                write_state(PUBLIC/'status.json',dict(status='WAITING_FOR_COMPLETE_CURVE_AND_PREFILL',assigned_chips=0));time.sleep(5)
            register(a.root,manifest,queue)
        engine.IDENTITY=IDENTITY;engine.PUBLIC=PUBLIC;engine.GuardedExecution=MixingExecution
        engine.run(a.root,resume=a.resume)
        file=a.root/IDENTITY/'report.json';result=json.loads(file.read_text())
        if result['status']!='COMPLETE_FIXED_TWO_NTP_TUNING':raise ValueError('Complete unchanged finite TUNE required')
        result.update(status='COMPLETE_FIXED_TWO_MIXING_CONTROL_TUNING',transformer_core_executions_per_window=1,
                      adapter_mixes_per_window=2,scientific_main_admitted=False)
        write_state(file,result);write_state(PUBLIC/'report.json',result);write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))
        write_state(a.root/QUEUE/'report.json',dict(status=result['status'],report_sha256=file_sha256(file),assigned_chips=0))
    except BaseException as error:
        receipt=dict(status='STOPPED_FIXED_TWO_MIXING_CONTROL_QUEUE',error=str(error),assigned_chips=0,
            native_plan_created=(a.root/IDENTITY/'plan.json').exists(),test_accessed=False,terminal_alpha_spent=0)
        previous=a.root/QUEUE/'report.json'
        if not previous.exists():write_state(previous,receipt)
        write_state(PUBLIC/'queue-failure.json',receipt)
        if not (PUBLIC/'report.json').exists():write_state(PUBLIC/'status.json',receipt)
        raise


if __name__=='__main__':main()
