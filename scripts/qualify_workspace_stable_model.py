"""Same fixed C0/saturated weights;stable FP32 loss versus independent FP64 loss."""
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
    from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig
    from compute_trainium.workspace_sharded_optimizer import WorkspaceShardedAdamW,ShardedGradientAccumulator
    from compute_trainium.workspace_native_objective import SelectiveWorkspacePredictor,drain_runtime
    from compute_trainium.workspace_native_state import restore_model_shards
    from compute_trainium.workspace_learning import WorkspaceFormat
    from compute_trainium.scale_model_sharding import shard_model_from_cpu
    from compute_trainium.scale_sharded_candidate import local_parameter
    plan=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or plan['purpose']!='workspace_stable_model_fixed_points' or plan['world']!=64 or plan['root_comm_ports'][a.output.name]!=a.comm_port:raise ValueError('Frozen fixed-point qualification required')
    if any(file_sha256(source/n)!=h for n,h in plan['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
    a.output.mkdir(parents=True,exist_ok=True)
    def progress(stage):
        if rank==0:(a.output/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started)))
    c0=torch.load(a.root/plan['checkpoint_relative'],weights_only=True,mmap=True)['model'];config=OlmoWorkspaceConfig(**plan['model_config'])
    with torch.device('meta'):model=WorkspaceOlmoCandidate(config,activation_checkpointing=True).eval()
    model=shard_model_from_cpu(model,mesh,c0,'neuron')
    with torch.device('meta'):scorer=WorkspaceOlmoCandidate(config).eval()
    scorer=shard_model_from_cpu(scorer,mesh,c0,'neuron').requires_grad_(False);del c0
    model.compile_stages('neuron');scorer.compile_stages('neuron')
    actor=SelectiveWorkspacePredictor(model);reader=SelectiveWorkspacePredictor(scorer);optimizer=WorkspaceShardedAdamW(model,fused=True)
    fmt=WorkspaceFormat(tuple(plan['opening_ids']),tuple(plan['closing_ids']));records={}
    for case in ['c0','mature']:
        progress(case+'_weights')
        if case=='mature':
            parent=a.root/plan['mature_shards_relative']/f'rank{rank}-u102.zip'
            restore_model_shards(parent,model,label='model');restore_model_shards(parent,scorer,label='scorer');drain_runtime(actor.device)
        inputs=torch.load(a.root/plan['cases'][case]['inputs_relative'],weights_only=True)
        c,y,z,rewards=[inputs[k] for k in ['prefix','targets','workspace','rewards']]
        versions=tuple(p._version for p in model.parameters())
        progress(case+'_frozen_scores');scores=reader.score(c,y,z,format=fmt)
        progress(case+'_full_stable_gradient');optimizer.zero_grad();accumulator=ShardedGradientAccumulator(model)
        losses=actor.backward(c,y,z,rewards,format=fmt,accumulator=accumulator,world=world)
        del accumulator;norm=optimizer.finish_gradients(scale=256,clip=1e30)
        expected=torch.load(a.root/plan['cases'][case]['gradient_relative'],weights_only=True,mmap=True)
        numbers=[];names=[]
        for name,p in model.named_parameters():
            wanted=expected[name].chunk(world,0)[rank].double();actual=local_parameter(p.grad).detach().cpu().double()
            numbers.append([float((actual-wanted).square().sum()),float(wanted.square().sum())]);names.append(name)
        values=torch.tensor(numbers,dtype=torch.float32,device='neuron');dist.all_reduce(values);values=values.cpu().double()
        errors=(values[:,0]/values[:,1].clamp_min(1e-30)).sqrt()
        if float(errors.max())>.02:raise AssertionError(case+' complete stable gradient exceeds unchanged2% named limit')
        if any(p.grad is not None for p in scorer.parameters()) or versions!=tuple(p._version for p in model.parameters()):raise AssertionError('Frozen scorer or zero-update actor changed')
        records[case]=dict(gradient_relative_l2=float((values[:,0].sum()/values[:,1].sum()).sqrt()),largest_named_relative_l2=float(errors.max()),
            largest_named_parameter=names[int(errors.argmax())],gradient_norm=norm,weighted_losses=losses,
            stable_absolute_gains=scores['absolute_gain'].tolist(),frozen_reward_labels=True)
        optimizer.zero_grad();del expected,numbers,values;drain_runtime(actor.device)
        if rank==0:(a.output/'case-results.json').write_text(json.dumps(records,indent=2)+'\n')
    fallback=torch_neuronx.get_fallback_ops()
    if fallback:raise AssertionError('Stable model used CPU fallback:'+str(fallback))
    dist.barrier()
    if rank==0:(a.output/'report.json').write_text(json.dumps(dict(status='PASS_WORKSPACE_STABLE_FIXED_MODEL_POINTS',world=world,records=records,
        wall_seconds=time.time()-started,optimizer_updates=0,fallback_ops=fallback,policy_fit_admitted=False,
        scope='Same unchanged C0 and saturated u102 checkpoint,two independent full179-parameter FP64-loss references;stable FP32 target loss on Native.No historical update is repaired or recertified.'),indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);run(p.parse_args())
