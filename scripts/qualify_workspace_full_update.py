"""One actual-model qualification update with streamed exact full-state restore."""
import argparse
import copy
import json
import os
from pathlib import Path

# Set core visibility before importing Torch through the shared entry.
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def post_gradient(model,scorer,optimizer,root,plan,out,progress):
    import torch
    import torch.distributed as dist
    from compute_trainium.scale_sharded_candidate import local_parameter
    from compute_trainium.workspace_sharded_optimizer import sharded_ema
    from compute_trainium.workspace_streaming_checkpoint import save_streaming,restore_streaming
    rank,world=dist.get_rank(),dist.get_world_size();progress('one_adamw_qualification_update')
    optimizer.finish_gradients(scale=1.,clip=1.)
    for group in optimizer.adam.param_groups:group['lr']=1e-5
    optimizer.step();sharded_ema(scorer,model,decay=.5)
    # Independent CPU first AdamW update from the HF gradient,one local matrix
    # at a time. No full CPU AdamW state is constructed on every rank.
    initial=torch.load(root/plan['checkpoint_relative'],weights_only=True,mmap=True)['model']
    reference=torch.load(root/plan['gradient_relative'],weights_only=True,mmap=True)
    norm=plan['independent_cpu_gradient_norm']
    records=[];names=[];scorer_named=dict(scorer.named_parameters())
    progress('independent_cpu_adamw_moments_delta_ema')
    for alias,(name,p) in zip(optimizer.local,optimizer.named,strict=True):
        start=initial[name].chunk(world,0)[rank] if p.placements[0].is_shard() else initial[name]
        gradient=reference[name].chunk(world,0)[rank] if p.placements[0].is_shard() else reference[name]
        parameter=torch.nn.Parameter(start.clone());parameter.grad=gradient.clone()*min(1.,1./(norm+1e-6))
        cpu=torch.optim.AdamW([parameter],lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False);cpu.step()
        actual=local_parameter(p).detach().cpu();expected=parameter.detach();state=optimizer.adam.state[alias]
        pairs=[(actual-start,expected-start),
               (state['exp_avg'].detach().cpu(),cpu.state[parameter]['exp_avg']),
               (state['exp_avg_sq'].detach().cpu(),cpu.state[parameter]['exp_avg_sq']),
               (local_parameter(scorer_named[name]).detach().cpu(),start*.5+expected*.5)]
        row=[]
        for actual_value,expected_value in pairs:
            row.extend([float((actual_value.double()-expected_value.double()).square().sum()),float(expected_value.double().square().sum())])
        records.append(row);names.append(name)
        del pairs,actual,expected,parameter,cpu
    value=torch.tensor(records,dtype=torch.float32,device='neuron');dist.all_reduce(value);value=value.cpu().double()
    errors={label:float((value[:,2*i]/value[:,2*i+1].clamp_min(1e-30)).sqrt().max()) for i,label in enumerate(['parameter_delta','adam_m','adam_v','ema_weights'])}
    numerical_pass=max(errors.values())<=.02
    (out/f'rank{rank}-update-audit.json').write_text(json.dumps(errors,indent=2)+'\n')
    optimizer.zero_grad();del initial,reference,records,value
    def tensors():
        for label,current in [('model',model),('scorer',scorer)]:
            for name,p in current.named_parameters():yield label+'.'+name,local_parameter(p)
        for index,alias in enumerate(optimizer.local):
            for name,value in sorted(optimizer.adam.state[alias].items()):
                if not isinstance(value,torch.Tensor):raise TypeError('Expected complete AdamW tensor state')
                yield f'adam.{index}.{name}',value
    generator=torch.Generator().manual_seed(415031+rank)
    groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.adam.param_groups]
    metadata=dict(plan_sha256=plan['registered_plan_sha256'],update=optimizer.update,parameter_names=[n for n,p in optimizer.named],
        param_groups=copy.deepcopy(groups),torch_rng=torch.get_rng_state(),sampling_rng=generator.get_state(),
        data_cursor=0,qualification_anchor_only=True,inputs_sha256=plan['inputs_sha256'])
    progress('streamed_full_model_optimizer_scorer_rng_save')
    checkpoint=out/f'rank{rank}-u1.zip';receipt=save_streaming(checkpoint,tensors(),metadata)
    with torch.no_grad():
        for name,value in tensors():value.reshape(-1)[0].add_(1.)
    optimizer.update+=1
    for group in optimizer.adam.param_groups:group['lr']=.123
    torch.rand(7);generator.manual_seed(1)
    progress('streamed_full_state_perturb_restore')
    restored=restore_streaming(checkpoint,tensors())
    if restored['parameter_names']!=metadata['parameter_names'] or restored['inputs_sha256']!=metadata['inputs_sha256']:raise AssertionError('Full recovery coverage/fingerprint')
    optimizer.update=restored['update']
    for group,values in zip(optimizer.adam.param_groups,restored['param_groups'],strict=True):group.update(values)
    torch.set_rng_state(restored['torch_rng']);generator.set_state(restored['sampling_rng'])
    if optimizer.update!=1 or not torch.equal(generator.get_state(),metadata['sampling_rng']) or not torch.equal(torch.get_rng_state(),metadata['torch_rng']):raise AssertionError('Full counters/RNG recovery differs')
    dist.barrier()
    if not numerical_pass:raise AssertionError('Actual-model AdamW deltas/moments/EMA mismatch;complete failed state was preserved')
    return dict(status='PASS_WORKSPACE_FULL_MODEL_UPDATE_AND_RESTORE',optimizer_updates=1,
        actual_qualification_policy_optimizer_updates=1,scientific_policy_fit_updates=0,actor_and_scorer_versions_unchanged=False,zero_update_versions_checked_before_the_qualification_update=True,
        complete_model_scorer_adamw_rng_restore_exact=True,checkpoint_bytes_per_rank=receipt['bytes'],
        checkpoint_tensor_count_per_rank=receipt['tensors'],independent_cpu_largest_named_errors=errors,
        fresh_process_next_update_replay=False,ema_decay_for_this_software_qualification=.5,
        scope='One real1.485B qualification update on the already consumed single TRAIN anchor.Independent full HF gradient,CPU AdamW deltas/moments,exact perturb/restore of every model/scorer/optimizer tensor and RNG.Future actual-size fresh-process trajectory and100warm-update throughput remain unqualified.No learned useful-thought,transfer or baseline-efficiency claim.')


if __name__=='__main__':
    from scripts.qualify_workspace_full_model import run
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True)
    run(p.parse_args(),post_gradient=post_gradient)
