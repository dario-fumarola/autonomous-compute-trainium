"""Bounded zero-update precision localization at a failed mature point."""
import argparse
import json
from pathlib import Path
import signal
import shutil
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.execution_isolation import managed_running
from scripts.run_scale_sharding_fixture import AllChipExecution,SCALE_END

IDENTITY='scale-trained-precision-probe-v1'
PUBLIC=Path('results/scale_trained_precision_probe_20261027')


def prepare(root):
    import torch
    from compute_trainium.scale_training_state import reconstruct_training_weights
    from scripts.train_scale_registered import lineage
    if managed_running():raise RuntimeError('Precision diagnosis follows Native closure')
    original=root/'scale-medium-tune-v1';plan_file=original/'plan.json';parent=json.loads(plan_file.read_text())
    if json.loads((original/'report.json').read_text())['status']!='STOPPED_SCALE_TUNE':raise ValueError('Preserved failed TUNE controller required')
    diagnosis=root/'scale-first-tune-gradient-diagnosis-v1/audit';observation=json.loads((diagnosis/'observation.json').read_text())
    if (observation['gradient']['passes_two_percent_gate'] or not observation['signature_equal']
            or not observation['source_cursor_equal'] or not observation['target_count_equal']
            or abs(observation['native_minus_cpu_nll'])>.003):raise ValueError('Exact localized gradient-only failure required')
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve all precision attempts')
    if time.time()+2700>SCALE_END:raise TimeoutError('Complete45-minute diagnostic allowance must fit original scale cap')
    if shutil.disk_usage(root).free<(35+7)*1024**3:raise RuntimeError('Complete7GiB diagnostic artifact allowance and protected35GiB reserve required')
    folder.mkdir();case=parent['cases']['fixed-two-lr0'];signature=lineage(file_sha256(plan_file),case)
    weights,rows=reconstruct_training_weights(root/case['artifact_directory'],299,signature,case['work_per_update'])
    initial=torch.load(root/case['initializer'],map_location='cpu',mmap=True,weights_only=False)
    state={**initial['model'],**weights};state['lm_head.weight']=state['transformer.wte.weight']
    point=folder/'unchanged-model.pt'
    torch.save(dict(purpose='ineligible_unchanged_trained_point_probe',eligible_for_science=False,
        original_study_sha256=file_sha256(plan_file),original_model_update=299,model=state),point)
    del weights,rows,initial,state
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()}
    for f in ['tests/test_scale_precision_probe.py','scripts/run_native_isolated.sh']:sources[f]=file_sha256(f)
    worker,host=split_snapshot_sources(sources);variants=['scaled16','boundary32','full32'];names=[IDENTITY+'-'+v for v in variants]
    start=time.time();plan=dict(identity=IDENTITY,purpose='unchanged_scale_point_precision_diagnosis_only',registered_unix=start,
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        world=64,worker_memory_gib=192,seconds_limit=2700,phase_seconds_limit=900,cpu_phase_seconds_limit=300,
        absolute_deadline_unix=min(start+2700,SCALE_END),scale_budget_deadline_unix=SCALE_END,
        phases={n:dict(variant=v) for n,v in zip(names,variants)},variants=variants,root_comm_ports=reserve_ports(root,names),
        point_file=IDENTITY+'/unchanged-model.pt',point_sha256=file_sha256(point),model=case['model'],parameters=case['parameters'],
        original_science_plan_sha256=file_sha256(plan_file),original_failure_sha256=file_sha256(original/'report.json'),
        diagnosis_observation_sha256=file_sha256(diagnosis/'observation.json'),
        cpu_gradient_file='scale-first-tune-gradient-diagnosis-v1/audit/independent-cpu-gradient.pt',
        cpu_gradient_sha256=file_sha256(diagnosis/'independent-cpu-gradient.pt'),
        original_gradient_file=case['artifact_directory']+'/gradient-u299.pt',
        source_cursor=299*512,noise_directory=case['artifact_directory'],data=parent['data'],artifact_directory=IDENTITY,
        actual_optimizer_updates=0,maximum_global_objective_windows_per_variant=512,quality_scoring=False,scientific_fit_admitted=False,
        scope='Three numerical interventions on the same unchanged weights,targets and noise:8x manual loss scale,FP32 injection/readout boundaries,full FP32 body. Compare all against the retained independent FP32 vector at the original2% family/overall and0.003 NLL limits. No training or retroactive certification of failed history. Only a bounded diagnosis; any repair needs qualification and fresh training.')
    PUBLIC.mkdir();write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    import torch
    from compute_trainium.parcae_training_candidate import gradient_comparison
    folder=root/IDENTITY;execution=AllChipExecution(root,folder/'plan.json',folder/'execution');records={}
    result=dict(status='STOPPED_SCALE_PRECISION_DIAGNOSIS',records=records,scientific_fit_admitted=False)
    def stop(number,frame):raise KeyboardInterrupt('Precision diagnosis interrupted')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        for variant in execution.plan['variants']:
            name=IDENTITY+'-'+variant
            options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
                '--plan-sha',execution.sha,'--variant',variant,'--comm-port',str(execution.plan['root_comm_ports'][name])]
            native=execution.native(name,'scripts.probe_scale_precision_native',options,'CAPTURED_UNCHANGED_SCALE_PRECISION_GRADIENT')
            expected_file=root/execution.plan['cpu_gradient_file'];actual_file=folder/(variant+'-gradient.pt')
            if file_sha256(expected_file)!=execution.plan['cpu_gradient_sha256'] or file_sha256(actual_file)!=native['gradient_sha256']:
                raise AssertionError('Frozen CPU/Native vectors changed')
            available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            if available<544*1024**3:raise RuntimeError('CPU vector comparison reserve plus32GiB headroom required')
            expected=torch.load(expected_file,map_location='cpu',mmap=True,weights_only=False)
            actual=torch.load(actual_file,map_location='cpu',mmap=True,weights_only=False)
            delta=gradient_comparison(expected['gradients'],actual['gradients']);difference=actual['loss']-expected['cpu_nll']
            row=dict(native=native,gradient=delta,native_minus_cpu_nll=difference,
                passes_unchanged_point_gate=delta['passes_two_percent_gate'] and abs(difference)<=.003,
                actual_optimizer_updates=0,eligible_for_science=False)
            records[variant]=row;del expected,actual;write_state(folder/'progress.json',result);write_state(PUBLIC/'progress.json',result)
        result.update(status='COMPLETE_UNCHANGED_SCALE_PRECISION_DIAGNOSIS',
            scope='Same-point numerical localization only. Passing a zero-update intervention does not repair the original training history or qualify a new scientific recipe.')
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(plan_sha256=execution.sha);execution.close(folder/'execution-costs.json')
        write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',result)
        for s,handler in handlers.items():signal.signal(s,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
