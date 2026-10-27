"""Independent upstream FP32 audit of quarantined large-shape Native captures."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def moments_from_shards(rows):
    import torch
    schema=rows[0]['optimizer']['schema'];result={}
    for offset,name in enumerate(schema['groups'][0]):
        pieces=[]
        for row in rows:
            adam=row['optimizer']['adamw'];slot=adam['param_groups'][0]['params'][offset]
            pieces.append(adam['state'][slot])
        step=int(pieces[0]['step'])
        if any(int(p['step'])!=step for p in pieces):raise AssertionError('AdamW steps differ across shards')
        item=dict(step=step)
        for source,destination in [('exp_avg','m'),('exp_avg_sq','v')]:
            tensors=[p[source] for p in pieces]
            if schema['placements'][name]=='S(0)':item[destination]=torch.cat(tensors,0)
            elif schema['placements'][name]=='R':
                if any(not torch.equal(tensors[0],t) for t in tensors[1:]):raise AssertionError('Replicated AdamW moments differ')
                item[destination]=tensors[0]
            else:raise ValueError('Unexpected moment placement')
        result[name]=item
    for index,names in enumerate(schema['groups'][1:]):
        owner=schema['owners'][index]
        state=rows[owner]['optimizer']['muon'][index]
        if any(index in row['optimizer']['muon'] for i,row in enumerate(rows) if i!=owner):
            raise AssertionError('Muon moment exists outside its owner')
        for offset,name in enumerate(names):
            result[name]=dict(m=state['m'][offset],v=state['v'][offset])
    return result


def comparison(before,expected,actual,groups):
    from compute_trainium.parcae_update_qualification import compare_update,compare_moments
    row=dict(update=compare_update(before,expected['model'],before,actual['model']),
        moments=compare_moments(expected['moments'],actual['moments']),families={})
    for name,names in [('adamw',groups[0]),('muon',[n for group in groups[1:] for n in group])]:
        row['families'][name]=dict(update=compare_update(before,expected['model'],before,actual['model'],names=names),
            moments=compare_moments({n:expected['moments'][n] for n in names},{n:actual['moments'][n] for n in names}))
    row['passes']=all(metric['relative_l2']<=.02 for part in [row,*row['families'].values()] for metric in [part['update'],part['moments']])
    return row


def audit(root,output,registration):
    started=time.time();output.mkdir(exist_ok=False)
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from compute_trainium.execution_isolation import managed_running
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_training_candidate import gradient_comparison,configure_optimizer
    from compute_trainium.parcae_optimizer_precise import independent_fp32_reference
    from compute_trainium.parcae_optimizer_alignment import align_reference_groups
    from compute_trainium.parcae_update_qualification import named_moments
    from compute_trainium.scale_model_state import reconstruct_weights
    from compute_trainium.training import PackedCorpus,file_sha256
    from scripts.audit_parcae_precise_training_point import make_reference
    from scripts.qualify_parcae_candidate_cpu import PIN
    from scripts.qualify_parcae_optimizer_cpu import unwrapped_reference
    from third_party.parcae_reference import muon_adamw as independent_optimizer
    from scripts.qualify_native import digest
    registered=json.loads(registration.read_text())
    code=Path(__file__).resolve().parents[1]
    if (registered['purpose']!='independent_scale_numerical_cpu_only' or registered['maximum_cpu_optimizer_steps']!=1
            or registered['maximum_global_objective_windows']!=512):raise ValueError('Exact bounded CPU registration required')
    if any(file_sha256(code/p)!=sha for p,sha in registered['source_sha256'].items()):
        raise AssertionError('Frozen independent CPU sources changed')
    if managed_running():raise RuntimeError('Large CPU audit must follow Native closure')
    torch.set_num_threads(24)
    folder=root/registered.get('native_identity','scale-956m-numerical-v1');plan_file=folder/'plan.json';plan=json.loads(plan_file.read_text())
    recurrent=plan.get('model_role','fixed_two')=='fixed_two'
    captured=json.loads((folder/'report.json').read_text())
    if (captured['status']!='CAPTURED_SCALE_ACTUAL_SHAPE_CPU_AUDITS_PENDING'
            or captured['plan_sha256']!=file_sha256(plan_file)):raise ValueError('Complete captured warm and replay required')
    if (file_sha256(plan_file)!=registered['native_plan_sha256']
            or file_sha256(folder/'report.json')!=registered['native_capture_report_sha256']):
        raise AssertionError('Frozen Native capture binding changed')
    artifact=Path(plan['host_artifact_root']);warm=artifact/'warm';replay=artifact/'replay'
    upstream=root/'upstream-parcae-69284c1'
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN:
        raise AssertionError('Pinned independent upstream changed')
    if subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip():
        raise AssertionError('Independent upstream must be clean')
    sys.path.insert(0,str(upstream));config=ParcaeCandidateConfig(**plan['model'])
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<608*1024**3:raise RuntimeError('Protected CPU audit reserve plus 96GiB headroom')
        if min(shutil.disk_usage(artifact).free,shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Protected CPU audit filesystem reserve')
        if time.time()-started>7200:raise TimeoutError('Original two-hour independent audit cap')
        if time.time()>registered['absolute_deadline_unix']:raise TimeoutError('Original registered CPU deadline')
    guard()
    initial_folder=root/'scale-medium-random-initializers-v1' if plan.get('size_profile')=='medium' else root.parent/'recurrence/scale-random-initializers-v1'
    initializer=initial_folder/('fixed_two.pt' if recurrent else 'ordinary8.pt')
    if file_sha256(initializer)!=plan['initializer_sha256']:raise AssertionError('Qualification initializer changed')
    initial=torch.load(initializer,map_location='cpu',mmap=True,weights_only=False)
    if recurrent:model=make_reference(config,True)
    else:
        from scripts.audit_ordinary8_native import make_reference as ordinary_reference
        model=ordinary_reference(config,False)
    model.load_state_dict(initial['model'],strict=True)
    first_weights,first_rows=reconstruct_weights(warm,1,config)
    first_vector=torch.load(warm/'gradient-u0.pt',map_location='cpu',mmap=True,weights_only=False)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    unwrapped_reference();precise=independent_fp32_reference(independent_optimizer)
    schema=first_rows[0]['optimizer']['schema']
    groups=independent_optimizer.get_muon_param_groups(model.named_parameters(),adamw_lr=.008,
        adamw_betas=(.8,.95),adamw_eps=1e-10,adamw_wd=0.,muon_lr=.008)
    groups=align_reference_groups(model,groups,schema['groups'])
    optimizer=independent_optimizer.MuonAdamW(groups)
    for name,p in model.named_parameters():p.grad=first_vector['gradients'][name].clone()
    configure_optimizer(optimizer,0)
    original=independent_optimizer.muon_step
    try:
        independent_optimizer.muon_step=precise;optimizer.step()
    finally:independent_optimizer.muon_step=original
    (output/'cpu-prescribed-step-completed.json').write_text(json.dumps(dict(prescribed_cpu_optimizer_steps=1))+'\n')
    first=comparison(before,dict(model={n:p.detach() for n,p in model.named_parameters()},
        moments=named_moments(model,optimizer,reference=True)),
        dict(model=first_weights,moments=moments_from_shards(first_rows)),schema['groups'])
    (output/'first-update.json').write_text(json.dumps(first,indent=2)+'\n')
    if not first['passes']:raise AssertionError('Independent first complete-matrix update/moment gate failed')
    del optimizer,first_rows,first_weights,first_vector,before;gc.collect();guard()
    weights,rows=reconstruct_weights(warm,100,config)
    state={**initial['model'],**weights};state['lm_head.weight']=state['transformer.wte.weight']
    model.load_state_dict(state,strict=True);model.zero_grad(set_to_none=True)
    model_digest=digest(dict(model.state_dict()));del state,weights,rows;gc.collect()
    packed=PackedCorpus.from_mapped(root/plan['data']['packed'],1024,23,plan['data']['source_sha256'])
    cursor=51200;targets=sum(packed.counts(cursor,512));total_loss=0.;current_state=None
    if recurrent:model.initialize_state=lambda _:current_state
    noises=[torch.load(warm/f'noise-u100-rank{i}.pt',map_location='cpu',mmap=True,weights_only=False) for i in range(64)] if recurrent else None
    with sdpa_kernel(SDPBackend.MATH):
        for index in range(512):
            guard();micro,rank=divmod(index,64);window=cursor+index
            if recurrent:
                if noises[rank]['windows'][micro]!=window:raise AssertionError('Explicit global objective state order differs')
                current_state=noises[rank]['states'][micro:micro+1].clone()
            batch=packed.batch([window],0)[0]
            kwargs={'num_steps_pair':torch.tensor([0,2])} if recurrent else {}
            logits=model(batch.inputs,return_logits=True,**kwargs)['logits']
            value=(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none')*batch.target_mask.flatten()).sum()
            (value/targets).backward();total_loss+=float(value.detach())
            if (index+1)%16==0:
                (output/'progress.json').write_text(json.dumps(dict(windows=index+1,total_windows=512,
                    elapsed_seconds=time.time()-started))+'\n')
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False))
    actual=torch.load(warm/'gradient-u100.pt',map_location='cpu',mmap=True,weights_only=False)
    delta=gradient_comparison({n:p.grad for n,p in model.named_parameters()},actual['gradients'])
    nll=total_loss/targets;difference=actual['loss']-nll
    if (not delta['passes_two_percent_gate'] or abs(difference)>.003 or actual['source_cursor']!=cursor
            or actual['real_targets']!=targets or not actual['clipped']):raise AssertionError('Full512 independent gradient/NLL gate failed')
    if digest(dict(model.state_dict()))!=model_digest:raise AssertionError('CPU backward changed checkpoint weights')
    gradient=dict(gradient=delta,cpu_nll=nll,native_minus_cpu_nll=difference,cpu_norm_before_clip=norm,
        native_norm_before_clip=actual['raw_gradient_norm'],windows=512,real_targets=targets,source_cursor=cursor)
    (output/'full-gradient.json').write_text(json.dumps(gradient,indent=2)+'\n')
    del model,actual,initial,noises;gc.collect();guard()
    before,rows=reconstruct_weights(warm,100,config);del rows
    wanted,expected_rows=reconstruct_weights(warm,101,config)
    seen,replay_rows=reconstruct_weights(replay,101,config)
    replay_result=comparison(before,dict(model=wanted,moments=moments_from_shards(expected_rows)),
        dict(model=seen,moments=moments_from_shards(replay_rows)),schema['groups'])
    if not replay_result['passes']:raise AssertionError('Independent named fresh-process replay gate failed')
    result=dict(status='PASS_SCALE_ACTUAL_SHAPE_INDEPENDENT_CPU_AUDITS',plan_sha256=file_sha256(plan_file),
        first_update=first,full512=gradient,replay=replay_result,reference_pin=PIN,world=64,
        scientific_fit_admitted=False,research_optimizer_updates=0,prescribed_cpu_optimizer_steps=1,
        wall_seconds=time.time()-started,scope='Independent numerical proof only, no held-out quality or efficiency claim.')
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--registration',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve every previous CPU audit')
    try:audit(args.root,args.output,args.registration)
    except BaseException as error:
        args.output.mkdir(exist_ok=True)
        (args.output/'failure.json').write_text(json.dumps(dict(status='STOPPED_SCALE_INDEPENDENT_CPU_AUDIT',error=str(error)))+'\n')
        raise
