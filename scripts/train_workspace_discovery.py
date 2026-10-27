"""One frozen, bounded discovery condition; no held-out reads during fitting."""
import argparse
import json
import math
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
    from compute_trainium.workspace_sharded_optimizer import WorkspaceShardedAdamW,ShardedGradientAccumulator,sharded_ema
    from compute_trainium.workspace_native_state import save_complete_state
    from compute_trainium.workspace_native_objective import SelectiveWorkspacePredictor,drain_runtime
    from compute_trainium.workspace_learning import WorkspaceFormat
    from compute_trainium.workspace_discovery import discovery_terms,backward_terms,step_matrix_work
    from compute_trainium.workspace_view_batches import WorkspaceViewBatches
    from compute_trainium.scale_model_sharding import shard_model_from_cpu
    from compute_trainium.scale_sharded_candidate import local_parameter
    plan=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or plan['purpose']!='workspace_four_arm_discovery_pilot' or plan['world']!=64 or plan['root_comm_ports'][a.output.name]!=a.comm_port:raise ValueError('Exact discovery registration required')
    if any(file_sha256(source/n)!=h for n,h in plan['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
    arm=a.arm;total=plan['updates'][arm];rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    def progress(stage,**extra):
        if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started,**extra)))
    progress('initializing_common_c0_and_fresh_optimizer',arm=arm)
    state=torch.load(a.root/plan['checkpoint_relative'],weights_only=True,mmap=True)['model'];config=OlmoWorkspaceConfig(**plan['model_config'])
    with torch.device('meta'):model=WorkspaceOlmoCandidate(config,activation_checkpointing=True).eval()
    model=shard_model_from_cpu(model,mesh,state,'neuron')
    scorer=None
    if arm=='workspace':
        with torch.device('meta'):scorer=WorkspaceOlmoCandidate(config).eval()
        scorer=shard_model_from_cpu(scorer,mesh,state,'neuron').requires_grad_(False)
    del state
    optimizer=WorkspaceShardedAdamW(model,lr=plan['lr'],fused=True);generator=torch.Generator().manual_seed(plan['sampling_seed_base']+rank)
    model.compile_stages('neuron')
    if scorer is not None:scorer.compile_stages('neuron')
    actor=SelectiveWorkspacePredictor(model,positions=160 if arm=='ordinary' else 192)
    reader=SelectiveWorkspacePredictor(scorer) if scorer is not None else None
    data=WorkspaceViewBatches(a.root/plan['data_relative'],'train')
    fmt=WorkspaceFormat(tuple(plan['opening_ids']),tuple(plan['closing_ids']));records=[]
    if arm=='ordinary':
        progress('new_160_position_full_gradient_gate',arm=arm)
        view=data.at(0);optimizer.zero_grad();accumulator=ShardedGradientAccumulator(model)
        backward_terms(actor,discovery_terms(arm,view['prefix'],view['continuation'],None,None,fmt,pause_id=plan['pause_token_id']),accumulator=accumulator,world=world)
        del accumulator;optimizer.finish_gradients(scale=256,clip=1e30)
        expected=torch.load(a.root/plan['ordinary_reference_relative'],weights_only=True,mmap=True)
        numbers=[]
        for name,p in model.named_parameters():
            wanted=expected[name].chunk(world,0)[rank].double();actual=local_parameter(p.grad).detach().cpu().double()
            numbers.append([float((actual-wanted).square().sum()),float(wanted.square().sum())])
        values=torch.tensor(numbers,dtype=torch.float32,device='neuron');dist.all_reduce(values);values=values.cpu().double()
        errors=(values[:,0]/values[:,1].clamp_min(1e-30)).sqrt()
        if float(errors.max())>.02:raise AssertionError('New ordinary context exceeds unchanged2% named gradient gate')
        if rank==0:(out/'initial-gradient-gate.json').write_text(json.dumps(dict(parameters=len(numbers),largest_named_relative_l2=float(errors.max()),optimizer_updates=0)))
        optimizer.zero_grad();del expected,numbers,values,errors;drain_runtime(actor.device)
    def objective(c,y,*,shared=False):
        z=rewards=None;scores=None
        if arm in ['workspace','draft']:
            z=actor.sample(c,format=fmt,length=8,group_size=4,generator=generator)
            if shared:
                value=z.to('neuron',dtype=torch.float32);dist.broadcast(value,src=0);z=value.cpu().long()
        if arm=='workspace':
            scores=reader.score(c,y[:,:4],z,format=fmt);rewards=scores['rewards']
            if shared:
                value=rewards.to('neuron');dist.broadcast(value,src=0);rewards=value.cpu()
        return z,rewards,scores
    for update in range(1,total+1):
        view=data.at((update-1)*world+rank);c,y=view['prefix'],view['continuation']
        progress('fitting',arm=arm,update=update,total_updates=total);tick=time.perf_counter()
        z,rewards,scores=objective(c,y)
        generation_scoring=time.perf_counter()-tick;tick=time.perf_counter();optimizer.zero_grad()
        accumulator=ShardedGradientAccumulator(model)
        losses=backward_terms(actor,discovery_terms(arm,c,y,z,rewards,fmt,pause_id=plan['pause_token_id']),accumulator=accumulator,world=world)
        del accumulator;norm=optimizer.finish_gradients(scale=256)
        gradient_time=time.perf_counter()-tick
        # This is a short constant-rate learning diagnostic, not a converged
        # or independently tuned efficiency endpoint. Keep the qualified fused
        # optimizer arithmetic unchanged; schedule qualification is separate.
        lr=plan['lr']
        tick=time.perf_counter();optimizer.step()
        if arm=='workspace':sharded_ema(scorer,model,decay=.99)
        drain_runtime(actor.device);optimizer_time=time.perf_counter()-tick
        row=dict(update=update,optimizer_counter=optimizer.update,lr=lr,generation_scoring_seconds=generation_scoring,
            gradient_seconds=gradient_time,optimizer_seconds=optimizer_time,total_update_seconds=generation_scoring+gradient_time+optimizer_time,
            gradient_norm=norm,weighted_losses=losses)
        if scores is not None:
            v=torch.tensor([float(scores['absolute_gain'].mean()),float(scores['absolute_gain'].std())],device='neuron');dist.all_reduce(v)
            v=(v/world).cpu().tolist();row.update(global_mean_absolute_gain=v[0],mean_within_anchor_gain_sd=v[1])
        records.append(row)
        if rank==0:(out/'updates.json').write_text(json.dumps(records,indent=2)+'\n')
        if not all(math.isfinite(v) for v in [norm,*losses.values()]):raise FloatingPointError('Nonfinite discovery update')
    optimizer.zero_grad();progress('mature_zero_update_independent_audit_inputs',arm=arm)
    view=data.at(plan['mature_audit_index']);c,y=view['prefix'],view['continuation']
    z,rewards,scores=objective(c,y,shared=True)
    accumulator=ShardedGradientAccumulator(model)
    losses=backward_terms(actor,discovery_terms(arm,c,y,z,rewards,fmt,pause_id=plan['pause_token_id']),accumulator=accumulator,world=world)
    del accumulator;norm=optimizer.finish_gradients(scale=256,clip=1e30)
    torch.save({n:local_parameter(p.grad).detach().cpu() for n,p in model.named_parameters()},out/f'rank{rank}-mature-gradients.pt')
    if rank==0:torch.save(dict(prefix=c,continuation=y,workspace=z,rewards=rewards),out/'private-mature-inputs.pt')
    optimizer.zero_grad();drain_runtime(actor.device);progress('saving_complete_endpoint',arm=arm)
    saved=save_complete_state(out/f'rank{rank}-u{total}.zip',model,scorer,optimizer,generator,
        extra=dict(plan_sha256=a.plan_sha,data_cursor=total*world,arm=arm,scientific_pilot=True,source_batches_sha256=plan['data_report_sha256']),compress=True)
    dist.barrier();fallback=torch_neuronx.get_fallback_ops()
    if fallback:raise AssertionError('Discovery used CPU fallback:'+str(fallback))
    if rank==0:
        warm=[r['total_update_seconds'] for r in records[1:]]
        result=dict(status='COMPLETE_WORKSPACE_DISCOVERY_ARM_PENDING_MATURE_AUDIT',arm=arm,world=world,
            actual_optimizer_updates=total,source_views=total*world,admitted_source_view_positions=total*world*160,
            consumed_source_positions=total*world*(160 if arm=='ordinary' else 136 if arm=='draft' else 132),
            natural_training_label_exposures=total*world*dict(ordinary=159,workspace=147,pause=135,draft=155)[arm],
            nominal_matrix_operations=total*world*step_matrix_work(config,arm),first_update_seconds=records[0]['total_update_seconds'],
            warm_updates=len(warm),warm_mean_seconds=sum(warm)/len(warm),wall_seconds=time.time()-started,
            actor_ledger=actor.ledger(),scorer_ledger=reader.ledger() if reader is not None else None,checkpoint_bytes_per_rank=saved['bytes'],
            fallback_ops=fallback,independent_mature_audit_pending=True,heldout_scores_read=False,
            scope='Frozen single-seed short discovery condition. Own prefix-only samples,no target-based evaluation selection,no teacher.No confirmation or from-scratch claim.Qualification and audit costs stay separate from retained-path nominal matrix work.')
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True)
    p.add_argument('--arm',choices=['ordinary','workspace','pause','draft'],required=True);run(p.parse_args())
