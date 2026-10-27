"""Ineligible62-update numerical reproduction; never replaces the missing failure."""
import argparse,json,os,shutil,time,subprocess
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from scripts.run_scale_sharding_fixture import AllChipExecution

IDENTITY='scale-recurrent-prefix-reproduction-v1';PUBLIC=Path('results/scale_recurrent_prefix_reproduction_20261027')


def prepare(root):
    import torch
    from scripts.qualify_native import digest
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every reproduction')
    original=root/'scale-medium-tune-65536-v1/plan.json';parent=json.loads(original.read_text());case=dict(parent['cases']['fixed-two-lr1'])
    if any(file_sha256(f)!=s for f,s in parent['source_sha256'].items()):raise AssertionError('Original listed mathematical sources unchanged required')
    initial_file=root/case['initializer']
    if file_sha256(initial_file)!=case['initializer_sha256']:raise AssertionError('Original unupdated C0 changed')
    if shutil.disk_usage(root).free<45*1024**3:raise RuntimeError('10GiB complete diagnostic state allowance plus35GiB reserve required')
    if time.time()+2700>parent['absolute_deadline_unix']:raise TimeoutError('Entire45-minute numerical allowance must fit original TUNE cutoff')
    initial=torch.load(initial_file,map_location='cpu',mmap=True,weights_only=False)
    if initial['optimizer_updates']!=0 or initial['model_forwards']!=0:raise ValueError('Only pristine initialization may be cloned')
    model_digest=digest(dict(initial['model']));folder.mkdir(mode=0o700);PUBLIC.mkdir()
    copied={**initial,'phase':'gate','eligible_for_science':False};target=folder/'ineligible-original-c0.pt';torch.save(copied,target)
    check=torch.load(target,map_location='cpu',mmap=True,weights_only=False)
    if digest(dict(check['model']))!=model_digest:raise AssertionError('Ineligible clone changed initial weights')
    del check,initial,copied
    case.update(phase='gate',initializer=IDENTITY+'/ineligible-original-c0.pt',initializer_sha256=file_sha256(target),
        lineage_case='reproduced-recurrent-lr1-prefix',endpoint_updates=300,stop_update=62,
        save_updates=[62],capture_before_updates=[],audit_first_update=False,post_training_probe=False,
        artifact_directory=IDENTITY+'/artifacts',retained_model=None)
    name=IDENTITY+'-warm';sources=dict(parent['source_sha256'])
    for f in ['scripts/reproduce_scale_recurrent_prefix.py','scripts/train_scale_capture_invalid_gradient.py']:sources[f]=file_sha256(f)
    worker,host=split_snapshot_sources(sources);start=time.time()
    prerequisite=root/'scale-ordinary-lr1-continuation-v1/plan.json'
    plan={**parent,'identity':IDENTITY,'registered_unix':start,'source_sha256':sources,'worker_source_sha256':worker,'host_only_source_sha256':host,
        'seconds_limit':2700,'absolute_deadline_unix':min(start+2700,parent['absolute_deadline_unix']),
        'phase_seconds_limit':900,'native_phase_seconds_limits':{name:900},'cpu_phase_seconds_limit':60,
        'invalid_gradient_capture':True,'cases':{'reproduce':case},'case_order':['reproduce'],
        'phases':{name:dict(case='reproduce')},'root_comm_ports':reserve_ports(root,[name]),
        'original_plan_sha256':file_sha256(original),'original_c0_sha256':parent['cases']['fixed-two-lr1']['initializer_sha256'],
        'unchanged_initial_parameter_digest':model_digest,'actual_optimizer_update_allowance':62,'quality_scoring':False,
        'original_failed_state_available':False,'reproduction_not_original_state':True,'scientific_main_fit_admitted':False,
        'prerequisite_ordinary_plan_sha256':file_sha256(prerequisite),'prerequisite_controller_pid':3890955,
        'scope':'Numerical reproduction only from an exact tensor clone of original zero-update C0,now gate-tagged ineligible. Original seed38213,state seed38246,source order,FP16/FP32 operators,65536 scale,0.016 rate and original300-update schedule stay fixed. Stop after62 updates before the missing failed step63;save complete actual state. No cooldown or held-out score. Scalar agreement cannot prove tensor identity to the unavailable original failure.45minutes includes waiting behind the complete ordinary cell; never renew the original TUNE or initial scale cutoff.'}
    write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    folder=root/IDENTITY;file=folder/'plan.json';plan=json.loads(file.read_text());sha=file_sha256(file)
    while True:
        if time.time()>plan['absolute_deadline_unix']:raise TimeoutError('Original diagnostic queue deadline')
        if file_sha256(file)!=sha or any(file_sha256(f)!=s for f,s in plan['source_sha256'].items()):raise AssertionError('Frozen numerical reproduction changed')
        parent=root/'scale-ordinary-lr1-continuation-v1/report.json'
        if parent.exists():
            receipt=json.loads(parent.read_text())
            if receipt['status']!='COMPLETE_ORIGINAL_REGISTERED_ORDINARY_LR1_CELL':raise RuntimeError('Ordinary cell did not close successfully; no automatic reproduction')
            break
        if subprocess.run(['kill','-0',str(plan['prerequisite_controller_pid'])],capture_output=True).returncode:raise RuntimeError('Prerequisite handle vanished without a terminal receipt')
        time.sleep(3)
    execution=AllChipExecution(root,file,folder/'execution');result=dict(status='STOPPED_NUMERICAL_PREFIX_REPRODUCTION',scientific_fit_admitted=False)
    try:
        name=IDENTITY+'-warm'
        options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json','--plan-sha',execution.sha,'--case','reproduce','--comm-port',str(plan['root_comm_ports'][name])]
        result['native']=execution.native(name,'scripts.train_scale_capture_invalid_gradient',options,'CAPTURED_SCALE_REGISTERED_TRAINING_PHASE')
        def rows(identity):return [json.loads(s) for s in (root/'runs'/identity/'updates.jsonl').read_text().splitlines()]
        original=rows('scale-medium-tune-65536-v1-fixed-two-lr1-warm');current=rows(name)
        if len(original)!=62 or len(current)!=62:raise AssertionError('Exactly62 original completed and reproduced steps required')
        result.update(status='CAPTURED_INELIGIBLE_REPRODUCED_PREFIX',all_scalar_losses_and_norms_equal=all(a['global_train_nll']==b['global_train_nll'] and a['gradient_norm']==b['gradient_norm'] for a,b in zip(original,current)),
            maximum_absolute_loss_difference=max(abs(a['global_train_nll']-b['global_train_nll']) for a,b in zip(original,current)),
            maximum_absolute_gradient_norm_difference=max(abs(a['gradient_norm']-b['gradient_norm']) for a,b in zip(original,current)),
            original_failed_state_available=False,scalar_agreement_not_full_tensor_identity=True,optimizer_updates=62,quality_scoring=False)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        execution.close(folder/'execution-costs.json');result['plan_sha256']=execution.sha;write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
