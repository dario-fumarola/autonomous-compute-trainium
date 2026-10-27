"""Independent upstream FP32 proof for a registered scale training endpoint."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def audit(root,plan_file,case_key,output):
    started=time.time();output.mkdir(exist_ok=False)
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from compute_trainium.training import file_sha256,PackedCorpus
    from compute_trainium.execution_isolation import managed_running
    from compute_trainium.scale_training_state import reconstruct_training_weights
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_training_candidate import gradient_comparison
    from compute_trainium.parcae_optimizer_alignment import align_reference_groups
    from compute_trainium.parcae_optimizer_precise import independent_fp32_reference
    from compute_trainium.parcae_update_qualification import named_moments
    from scripts.train_scale_registered import lineage
    from scripts.audit_scale_model import comparison,moments_from_shards
    from scripts.audit_parcae_precise_training_point import make_reference
    from scripts.qualify_parcae_candidate_cpu import PIN
    from scripts.qualify_parcae_optimizer_cpu import unwrapped_reference
    from scripts.qualify_native import digest
    from third_party.parcae_reference import muon_adamw as independent_optimizer
    plan=json.loads(plan_file.read_text());sha=file_sha256(plan_file);case=plan['cases'][case_key]
    if plan['purpose']!='scale_registered_development_v1':raise ValueError('Frozen scale training registration required')
    code=Path(__file__).resolve().parents[1]
    if any(file_sha256(code/f)!=s for f,s in plan['source_sha256'].items()):raise AssertionError('Frozen independent audit sources changed')
    if managed_running():raise RuntimeError('Independent dense CPU audit follows Native closure')
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<608*1024**3:raise RuntimeError('CPU audit protected512GiB plus96GiB headroom')
        if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('CPU audit storage reserve')
        if time.time()>min(started+plan['cpu_phase_seconds_limit'],plan['absolute_deadline_unix']):raise TimeoutError('Original CPU/study cap')
    guard();torch.set_num_threads(24)
    upstream=root/'upstream-parcae-69284c1'
    if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
            or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):
        raise AssertionError('Clean independent upstream pin required')
    sys.path.insert(0,str(upstream));config=ParcaeCandidateConfig(**case['model']);recurrent=case['model_role']=='fixed_two'
    signature=lineage(sha,case);folder=root/case['artifact_directory'];end=case['endpoint_updates'];per=case['work_per_update']
    def reconstruct(directory,update):
        weights,rows=reconstruct_training_weights(directory,update,signature,per)
        # Preserve the qualified independent named-moment extraction unchanged.
        return weights,[{**r,'optimizer':r['optimizer']['execution']} for r in rows]
    initial_file=root/case['initializer']
    if file_sha256(initial_file)!=case['initializer_sha256']:raise AssertionError('Original C0 changed')
    initial=torch.load(initial_file,map_location='cpu',mmap=True,weights_only=False)
    if recurrent:model=make_reference(config,True)
    else:
        from scripts.audit_ordinary8_native import make_reference as ordinary_reference
        model=ordinary_reference(config,False)
    model.load_state_dict(initial['model'],strict=True)
    first=None
    if case.get('audit_first_update',False):
        first_weights,first_rows=reconstruct(folder,1)
        first_vector=torch.load(folder/'gradient-u0.pt',map_location='cpu',mmap=True,weights_only=False)
        before={n:p.detach().clone() for n,p in model.named_parameters()};unwrapped_reference()
        groups=independent_optimizer.get_muon_param_groups(model.named_parameters(),adamw_lr=case['peak_lr'],
            adamw_betas=(.8,.95),adamw_eps=1e-10,adamw_wd=0.,muon_lr=case['peak_lr'])
        schema=first_rows[0]['optimizer']['schema'];groups=align_reference_groups(model,groups,schema['groups'])
        optimizer=independent_optimizer.MuonAdamW(groups)
        actual_groups=first_rows[0]['optimizer']['groups']
        for group,values in zip(optimizer.param_groups,actual_groups,strict=True):
            for key in ['lr','momentum','weight_decay','beta2','ns_steps']:
                if key in values:group[key]=values[key]
        for name,p in model.named_parameters():p.grad=first_vector['gradients'][name].clone()
        precise=independent_fp32_reference(independent_optimizer);original=independent_optimizer.muon_step
        try:independent_optimizer.muon_step=precise;optimizer.step()
        finally:independent_optimizer.muon_step=original
        first=comparison(before,dict(model={n:p.detach() for n,p in model.named_parameters()},
            moments=named_moments(model,optimizer,reference=True)),dict(model=first_weights,moments=moments_from_shards(first_rows)),schema['groups'])
        (output/'first-update.json').write_text(json.dumps(first,indent=2)+'\n')
        if not first['passes']:raise AssertionError('Independent first prescribed update/moments failed')
        del optimizer,first_weights,first_rows,first_vector,before;gc.collect();guard()
    point=end-1;weights,rows=reconstruct(folder,point);schema=rows[0]['optimizer']['schema']
    state={**initial['model'],**weights};state['lm_head.weight']=state['transformer.wte.weight']
    model.load_state_dict(state,strict=True);model.zero_grad(set_to_none=True);model_digest=digest(dict(model.state_dict()))
    del state,weights,rows;gc.collect();packed=PackedCorpus.from_mapped(root/plan['data']['packed'],1024,23,plan['data']['source_sha256'])
    cursor=point*512;targets=sum(packed.counts(cursor,512));noises={}
    if recurrent:
        for rank in range(64):
            row=torch.load(folder/f'noise-u{point}-rank{rank}.pt',map_location='cpu',mmap=True,weights_only=False)
            expected=[cursor+i*64+rank for i in range(8)]
            if row['windows']!=expected:raise AssertionError('Captured recurrent windows differ')
            noises.update(dict(zip(row['windows'],row['states'].split(1),strict=True)))
        current=None;model.initialize_state=lambda _:current
    total=0.
    with sdpa_kernel(SDPBackend.MATH):
        for offset in range(512):
            guard();window=cursor+offset;batch=packed.batch([window],0)[0]
            current=noises[window] if recurrent else None
            kwargs={'num_steps_pair':torch.tensor([0,2])} if recurrent else {}
            logits=model(batch.inputs,return_logits=True,**kwargs)['logits']
            value=(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none')*batch.target_mask.flatten()).sum()
            (value/targets).backward();total+=float(value.detach())
            if (offset+1)%16==0:(output/'progress.json').write_text(json.dumps(dict(windows=offset+1,total_windows=512,elapsed_seconds=time.time()-started))+'\n')
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False))
    actual=torch.load(folder/f'gradient-u{point}.pt',map_location='cpu',mmap=True,weights_only=False)
    delta=gradient_comparison({n:p.grad for n,p in model.named_parameters()},actual['gradients']);difference=actual['loss']-total/targets
    if (not delta['passes_two_percent_gate'] or abs(difference)>.003 or actual['signature']!=signature
            or actual['source_cursor']!=cursor or actual['real_targets']!=targets or not actual['clipped']):
        raise AssertionError('Independent complete512 gradient/NLL failed')
    if digest(dict(model.state_dict()))!=model_digest:raise AssertionError('Independent backward changed weights')
    gradient=dict(gradient=delta,native_minus_cpu_nll=difference,cpu_nll=total/targets,cpu_norm_before_clip=norm,windows=512,source_cursor=cursor,real_targets=targets)
    (output/'full-gradient.json').write_text(json.dumps(gradient,indent=2)+'\n')
    del model,actual,initial,noises;gc.collect();guard()
    replay=None
    if case.get('replay_directory'):
        before,rows=reconstruct(folder,point);del rows
        expected,expected_rows=reconstruct(folder,end);seen,replay_rows=reconstruct(root/case['replay_directory'],end)
        replay=comparison(before,dict(model=expected,moments=moments_from_shards(expected_rows)),
            dict(model=seen,moments=moments_from_shards(replay_rows)),schema['groups'])
        if not replay['passes']:raise AssertionError('Independent named replay update/moments failed')
    report=dict(status='PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT',case=case_key,phase=case['phase'],
        study_sha256=sha,first_update=first,full512=gradient,replay=replay,reference_pin=PIN,
        research_optimizer_updates=0,prescribed_cpu_optimizer_steps=1 if first is not None else 0,assigned_chips=0,
        wall_seconds=time.time()-started,scope='Independent numerical audit; no prediction-efficiency or confirmation claim.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:audit(a.root,a.plan,a.case,a.output)
    except BaseException as error:
        a.output.mkdir(exist_ok=True);(a.output/'failure.json').write_text(json.dumps(dict(status='STOPPED_SCALE_TRAINING_AUDIT',error_type=type(error).__name__,error=str(error)))+'\n');raise
