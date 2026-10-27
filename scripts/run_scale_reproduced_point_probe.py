"""Same reproduced point; high/low loss scales and a complete independent reference."""
import argparse,json,shutil,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from scripts.run_scale_sharding_fixture import AllChipExecution

IDENTITY='scale-reproduced-point-probe-v1';PUBLIC=Path('results/scale_reproduced_point_probe_20261027')


def prepare(root):
    import torch
    from compute_trainium.scale_training_state import reconstruct_training_weights
    from scripts.train_scale_registered import lineage
    repro=root/'scale-recurrent-prefix-reproduction-v1';source_plan=repro/'plan.json';parent=json.loads(source_plan.read_text());receipt=json.loads((repro/'report.json').read_text())
    if receipt['status']!='CAPTURED_INELIGIBLE_REPRODUCED_PREFIX' or not receipt['all_scalar_losses_and_norms_equal']:raise ValueError('Complete reproduced prefix scalar agreement required')
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve all point probes')
    if time.time()+1200>parent['absolute_deadline_unix']:raise TimeoutError('Complete20-minute probe/reference allowance must fit existing reproduction cap')
    if shutil.disk_usage(root).free<47*1024**3:raise RuntimeError('12GiB complete point,noise,Native/CPU vector allowance plus35GiB reserve required')
    mem=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if mem<704*1024**3:raise RuntimeError('Complete192GiB worker plus512GiB protected host reserve')
    folder.mkdir(mode=0o700);PUBLIC.mkdir();case=parent['cases']['reproduce']
    weights,rows=reconstruct_training_weights(root/case['artifact_directory'],62,lineage(file_sha256(source_plan),case),case['work_per_update'])
    initial=torch.load(root/case['initializer'],map_location='cpu',mmap=True,weights_only=False)
    state={**initial['model'],**weights};state['lm_head.weight']=state['transformer.wte.weight']
    point=folder/'reproduced-point.pt';torch.save(dict(purpose='ineligible_unchanged_trained_point_probe',eligible_for_science=False,
        reproduced_model_update=62,original_failed_state_available=False,model=state),point)
    del initial,state,weights,rows
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()}
    sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh')
    worker,host=split_snapshot_sources(sources);names=[IDENTITY+'-high',IDENTITY+'-low']
    plan=dict(identity=IDENTITY,purpose='reproduced_point_loss_scale_diagnosis_only',registered_unix=time.time(),
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        world=64,worker_memory_gib=192,seconds_limit=1200,phase_seconds_limit=600,cpu_phase_seconds_limit=1000,
        absolute_deadline_unix=parent['absolute_deadline_unix'],scale_budget_deadline_unix=parent['scale_budget_deadline_unix'],
        phases={n:dict(variant=v) for n,v in zip(names,['high','low'])},root_comm_ports=reserve_ports(root,names),
        variants={'high':{'manual_loss_scale':65536},'low':{'manual_loss_scale':8192}},capture_expected_nonfinite=True,
        point_file=IDENTITY+'/reproduced-point.pt',point_sha256=file_sha256(point),model=case['model'],parameters=case['parameters'],
        original_reproduction_plan_sha256=file_sha256(source_plan),reproduced_prefix_receipt_sha256=file_sha256(repro/'report.json'),
        source_cursor=62*512,state_seed=case['state_seed'],noise_directory=IDENTITY+'/native-noise',data=parent['data'],artifact_directory=IDENTITY,
        optimizer_updates=0,scientific_fit_admitted=False,quality_scoring=False,original_failed_state_available=False,
        scope='Fixed reproduced point62 and512 authentic TRAIN windows only. Same FP16/FP32 graph and immutable Native-generated explicit noise at both scales; parameters checked unchanged. Expected nonfinite gradient/norm is observed and retained,never treated as passing qualification. Full independent FP32 forward/backward withfloat64 norm follows Native closure. All work remains ineligible;no old failure is certified. The original reproduction10:46:44 deadline and initial scale cutoff remain unchanged.')
    write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    import torch
    from compute_trainium.parcae_training_candidate import gradient_comparison
    folder=root/IDENTITY;execution=AllChipExecution(root,folder/'plan.json',folder/'execution');records={};result=dict(status='STOPPED_REPRODUCED_POINT_PROBE',records=records,scientific_fit_admitted=False)
    try:
        for variant in ['high','low']:
            name=IDENTITY+'-'+variant
            opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--variant',variant,'--comm-port',str(execution.plan['root_comm_ports'][name])]
            records[variant]=execution.native(name,'scripts.probe_scale_loss_scale_native',opts,'CAPTURED_REPRODUCED_POINT_SCALE_OBSERVATION')
            write_state(folder/'progress.json',result)
        result['cpu']=execution.cpu('independent-reference','scripts.reference_scale_reproduced_point',['--root',str(root),'--plan',str(folder/'plan.json'),'--output',str(folder/'cpu-reference')],folder/'cpu-reference/report.json','CAPTURED_INDEPENDENT_REPRODUCED_POINT_FP32_REFERENCE')
        expected=torch.load(folder/'cpu-reference/reference-gradient.pt',map_location='cpu',mmap=True,weights_only=False)
        comparisons={}
        for variant in ['high','low']:
            actual=torch.load(folder/(variant+'-gradient.pt'),map_location='cpu',mmap=True,weights_only=False)
            row=dict(native_gradient_finite=actual['finite'],native_global_norm_finite=actual['global_norm_finite'],
                cpu_gradient_finite=expected['finite'],native_minus_cpu_nll=actual['loss']-expected['cpu_nll'])
            if expected['finite'] and actual['finite'] and actual['global_norm_finite']:
                row['gradient']=gradient_comparison(expected['gradients'],actual['gradients'])
                row['passes_same_point_numerical_thresholds']=row['gradient']['passes_two_percent_gate'] and abs(row['native_minus_cpu_nll'])<=.003
            else:row['passes_same_point_numerical_thresholds']=False
            comparisons[variant]=row
        result.update(status='COMPLETE_REPRODUCED_POINT_SCALE_DIAGNOSIS',comparisons=comparisons,
            loss_scale_only_determined_nonfiniteness_at_this_point=not records['high']['global_norm_finite'] and records['low']['global_norm_finite'] and comparisons['low']['passes_same_point_numerical_thresholds'],
            original_failed_state_available=False,optimizer_updates=0,no_training_history_repaired=True)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        execution.close(folder/'execution-costs.json');result['plan_sha256']=execution.sha;write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
