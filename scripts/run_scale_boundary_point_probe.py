"""One conditional zero-update boundary probe; same point/noise/reference/deadline."""
import argparse,json,shutil,subprocess,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from scripts.run_scale_sharding_fixture import AllChipExecution

IDENTITY='scale-reproduced-boundary-probe-v1';PUBLIC=Path('results/scale_reproduced_boundary_probe_20261027')


def prepare(root):
    parent_file=root/'scale-reproduced-point-probe-v1/plan.json';parent=json.loads(parent_file.read_text());folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every conditional diagnostic')
    finite_file=root/'scale-reproduced-point-probe-v1/parameter-finiteness.json'
    if not json.loads(finite_file.read_text())['all_finite']:raise ValueError('Boundary test requires finite saved parameters/buffers')
    for variant in ['high','low']:
        report=json.loads((root/'runs'/('scale-reproduced-point-probe-v1-'+variant)/'report.json').read_text())
        if report['global_norm_finite'] or report['gradient_finite']:raise ValueError('Both same-policy scales must have observed nonfiniteness')
    if time.time()+300>parent['absolute_deadline_unix']:raise TimeoutError('Complete five-minute Native/compare allowance must fit existing point cap')
    if shutil.disk_usage(root).free<38*1024**3:raise RuntimeError('Complete3GiB extra vector allowance plus protected35GiB required')
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()};sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh')
    worker,host=split_snapshot_sources(sources);name=IDENTITY+'-boundary'
    plan={**parent,'identity':IDENTITY,'registered_unix':time.time(),'source_sha256':sources,'worker_source_sha256':worker,'host_only_source_sha256':host,
        'seconds_limit':600,'phase_seconds_limit':300,'phases':{name:dict(variant='boundary')},'root_comm_ports':reserve_ports(root,[name]),
        'variants':{'boundary':{'manual_loss_scale':8192}},'artifact_directory':IDENTITY,
        'source_parent_plan_sha256':file_sha256(parent_file),'parameter_finiteness_sha256':file_sha256(finite_file),
        'prerequisite_parent_pid':3959144,'scope':'Conditional numerical diagnosis only after both loss scales give nonfinite results with finite saved parameters. Same immutable reproduced point,Native-generated noise and full independent CPU reference. FP32 injection/readout/residual boundaries,other bodyoperatorsFP16,scale8192. Zero optimizer updates. Does not certify old history or a new recipe. Original10:46:44 point cap is unchanged;activate only if the full FP32 CPU gradient is finite.'}
    folder.mkdir(mode=0o700);PUBLIC.mkdir();write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    import torch
    from compute_trainium.parcae_training_candidate import gradient_comparison
    folder=root/IDENTITY;plan_file=folder/'plan.json';plan=json.loads(plan_file.read_text());sha=file_sha256(plan_file);parent=root/'scale-reproduced-point-probe-v1'
    while not (parent/'report.json').exists():
        if time.time()+300>plan['absolute_deadline_unix']:raise TimeoutError('Complete boundary allowance no longer fits original cap')
        if subprocess.run(['kill','-0',str(plan['prerequisite_parent_pid'])],capture_output=True).returncode:raise RuntimeError('Parent stopped without a receipt')
        time.sleep(2)
    original=json.loads((parent/'report.json').read_text())
    if original['status']!='COMPLETE_REPRODUCED_POINT_SCALE_DIAGNOSIS' or not original['cpu']['gradient_finite']:raise RuntimeError('No finite full FP32 reference; boundary probe not admitted')
    execution=AllChipExecution(root,plan_file,folder/'execution');result=dict(status='STOPPED_REPRODUCED_BOUNDARY_DIAGNOSIS',optimizer_updates=0,scientific_fit_admitted=False)
    try:
        name=IDENTITY+'-boundary';options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--variant','boundary','--comm-port',str(plan['root_comm_ports'][name])]
        native=execution.native(name,'scripts.probe_scale_boundary_native',options,'CAPTURED_REPRODUCED_POINT_SCALE_OBSERVATION');result['native']=native
        expected=torch.load(parent/'cpu-reference/reference-gradient.pt',map_location='cpu',mmap=True,weights_only=False);actual=torch.load(folder/'boundary-gradient.pt',map_location='cpu',mmap=True,weights_only=False)
        row=dict(native_gradient_finite=actual['finite'],global_norm_finite=actual['global_norm_finite'],native_minus_cpu_nll=actual['loss']-expected['cpu_nll'],passes_original_thresholds=False)
        if expected['finite'] and actual['finite'] and actual['global_norm_finite']:
            row['gradient']=gradient_comparison(expected['gradients'],actual['gradients']);row['passes_original_thresholds']=row['gradient']['passes_two_percent_gate'] and abs(row['native_minus_cpu_nll'])<=.003
        result.update(status='COMPLETE_REPRODUCED_BOUNDARY_DIAGNOSIS',comparison=row,original_failed_state_available=False,no_training_history_repaired=True)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        execution.close(folder/'execution-costs.json');result['plan_sha256']=sha;write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
