"""Zero-update all64-rank actual1.485B workspace gradient qualification."""
import argparse
import json
import os
from pathlib import Path
import time

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def run(a):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from compute_trainium.training import file_sha256
    from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig,rotary_tensors
    from compute_trainium.workspace_sharded_optimizer import WorkspaceShardedAdamW,ShardedGradientAccumulator
    from compute_trainium.scale_model_sharding import shard_model_from_cpu
    from compute_trainium.scale_sharded_candidate import local_parameter
    from compute_trainium.workspace_joint_terms import workspace_joint_terms,padded_prediction_inputs
    from compute_trainium.workspace_learning import WorkspaceFormat,identity_reader_context
    root=a.root;plan=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if (file_sha256(a.plan)!=a.plan_sha or plan['purpose']!='workspace_actual_model_zero_update_gradient'
            or plan['world']!=64 or plan['root_comm_ports'][a.output.name]!=a.comm_port):raise ValueError('Frozen actual-model qualification plan required')
    if any(file_sha256(source/n)!=h for n,h in plan['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world)
    mesh=init_device_mesh('neuron',(world,));out=a.output;out.mkdir(parents=True,exist_ok=True)
    def progress(stage):
        if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started)))
    progress('loading_full_parameter_shards')
    c0=torch.load(root/plan['checkpoint_relative'],weights_only=True,mmap=True,map_location='cpu')
    config=OlmoWorkspaceConfig(**plan['model_config'])
    with torch.device('meta'):model=WorkspaceOlmoCandidate(config,activation_checkpointing=True).eval()
    model=shard_model_from_cpu(model,mesh,c0['model'],'neuron')
    with torch.device('meta'):scorer=WorkspaceOlmoCandidate(config).eval()
    scorer=shard_model_from_cpu(scorer,mesh,c0['model'],'neuron').requires_grad_(False)
    del c0
    model.compile_stages('neuron');scorer.compile_stages('neuron')
    optimizer=WorkspaceShardedAdamW(model,fused=True)
    inputs=torch.load(root/plan['inputs_relative'],weights_only=True)
    c,y,z=inputs['prefix'],inputs['targets'],inputs['workspace'];rewards=inputs['rewards']
    fmt=WorkspaceFormat(tuple(plan['opening_ids']),tuple(plan['closing_ids']))
    cos,sin=rotary_tensors(config,plan['padded_positions'],'neuron')
    actor_versions=tuple(p._version for p in model.parameters());scorer_versions=tuple(p._version for p in scorer.parameters())
    @torch.no_grad()
    def prediction(context,targets):
        padded,positions=padded_prediction_inputs(context,targets,plan['padded_positions'],config.pad_token_id)
        logits=scorer(padded.to('neuron'),cos,sin,positions.to('neuron'))
        value=torch.log_softmax(logits.float(),-1).gather(-1,targets.to('neuron').unsqueeze(-1)).squeeze(-1)
        return float(value.mean().cpu())
    progress('frozen_counterfactual_scores')
    raw=prediction(c,y);gains=[]
    for i in range(z.shape[0]):gains.append(prediction(identity_reader_context(c,z[i:i+1],fmt),y)-raw)
    expected_rewards=rewards+plan['cost_per_token']*z.shape[1]
    if max(abs(x-float(y)) for x,y in zip(gains,expected_rewards))>.002:raise AssertionError('Actual frozen-reader absolute gains differ from independent CPU')
    losses={};optimizer.zero_grad();accumulator=ShardedGradientAccumulator(model)
    for index,(phase,context,target,weight,reduction) in enumerate(workspace_joint_terms(c,y,z,rewards,fmt)):
        progress('joint_forward_backward_'+str(index+1)+'_'+phase)
        padded,positions=padded_prediction_inputs(context,target,plan['padded_positions'],config.pad_token_id)
        logits=model(padded.to('neuron'),cos,sin,positions.to('neuron'))
        nll=-torch.log_softmax(logits.float(),-1).gather(-1,target.to('neuron').unsqueeze(-1)).squeeze(-1)
        loss=weight*(nll.mean() if reduction=='mean' else nll.sum())
        losses[phase]=losses.get(phase,0.)+float(loss.detach().cpu())
        (loss/world*256).backward();accumulator.add_and_clear()
        del loss,nll,logits
    progress('full_named_gradient_comparison')
    accumulator.install();norm=optimizer.finish_gradients(scale=256,clip=1e30)
    expected=torch.load(root/plan['gradient_relative'],weights_only=True,mmap=True,map_location='cpu')
    numbers=[];names=[]
    for name,p in model.named_parameters():
        wanted=expected[name].chunk(world,0)[rank] if p.placements[0].is_shard() else expected[name]
        actual=local_parameter(p.grad).detach().cpu()
        factor=1. if p.placements[0].is_shard() else 1./world
        diff=actual.double()-wanted.double()
        numbers.append([float(diff.square().sum())*factor,float(wanted.double().square().sum())*factor])
        names.append(name)
    values=torch.tensor(numbers,dtype=torch.float32,device='neuron');dist.all_reduce(values);values=values.cpu().double()
    named=(values[:,0]/values[:,1].clamp_min(1e-30)).sqrt();overall=float((values[:,0].sum()/values[:,1].sum()).sqrt())
    if float(named.max())>.02:raise AssertionError('Full actual-model named gradient family exceeds2%')
    if any(p.grad is not None for p in scorer.parameters()) or actor_versions!=tuple(p._version for p in model.parameters()) or scorer_versions!=tuple(p._version for p in scorer.parameters()):raise AssertionError('Frozen scorer or zero-update actor changed')
    dist.barrier()
    if rank==0:
        result=dict(status='PASS_WORKSPACE_ACTUAL_MODEL_GRADIENT',world=world,parameters=sum(p.numel() for p in model.parameters()),
            wall_seconds=time.time()-started,gradient_norm=norm,gradient_relative_l2=overall,
            largest_named_relative_l2=float(named.max()),largest_named_parameter=names[int(named.argmax())],
            absolute_gains=gains,weighted_losses=losses,actual_joint_backward_calls=9,optimizer_updates=0,
            scorer_gradients_absent=True,actor_and_scorer_versions_unchanged=True,
            padded_positions=plan['padded_positions'],no_fallback=True,
            scope='Zero-update actual-model numerical qualification on one previously consumed TRAIN anchor.64 identical replicas produce one normalized objective,not64 unique examples. No policy fit,throughput or efficiency claim.')
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True)
    run(p.parse_args())
