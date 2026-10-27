"""One cold+100warm actual-size online updates;repeated TRAIN qualification only."""
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
    from compute_trainium.workspace_sharded_optimizer import WorkspaceShardedAdamW,ShardedGradientAccumulator,sharded_ema
    from compute_trainium.workspace_native_state import restore_complete_state,save_complete_state
    from compute_trainium.workspace_native_objective import SelectiveWorkspacePredictor,drain_runtime
    from compute_trainium.workspace_learning import WorkspaceFormat
    from compute_trainium.scale_model_sharding import shard_model_from_cpu
    from compute_trainium.scale_sharded_candidate import local_parameter
    plan=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or plan['purpose']!='workspace_stable_online_fresh_source_qualification' or plan['world']!=64 or plan['root_comm_ports'][a.output.name]!=a.comm_port:raise ValueError('Exact online qualification registration required')
    if any(file_sha256(source/n)!=h for n,h in plan['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    def progress(stage,**extra):
        if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started,**extra)))
    progress('initializing_fresh_process_full_shards')
    c0=torch.load(a.root/plan['checkpoint_relative'],weights_only=True,mmap=True)['model'];config=OlmoWorkspaceConfig(**plan['model_config'])
    with torch.device('meta'):model=WorkspaceOlmoCandidate(config,activation_checkpointing=True).eval()
    model=shard_model_from_cpu(model,mesh,c0,'neuron')
    with torch.device('meta'):scorer=WorkspaceOlmoCandidate(config).eval()
    scorer=shard_model_from_cpu(scorer,mesh,c0,'neuron').requires_grad_(False);del c0
    optimizer=WorkspaceShardedAdamW(model,lr=1e-5,fused=True);generator=torch.Generator()
    generator.manual_seed(7351+rank)
    model.compile_stages('neuron');scorer.compile_stages('neuron')
    actor=SelectiveWorkspacePredictor(model);reader=SelectiveWorkspacePredictor(scorer)
    from compute_trainium.workspace_view_batches import WorkspaceViewBatches
    data=WorkspaceViewBatches(a.root/plan['data_relative'],'train')
    fmt=WorkspaceFormat(tuple(plan['opening_ids']),tuple(plan['closing_ids']));records=[]
    for update in range(1,102):
        view=data.at((update-1)*world+rank);c,y=view['prefix'],view['continuation'][:,:4]
        progress('online_sampling',update=update);tick=time.perf_counter()
        z=actor.sample(c,format=fmt,length=8,group_size=4,generator=generator);sampling=time.perf_counter()-tick
        tick=time.perf_counter();scores=reader.score(c,y,z,format=fmt);scoring=time.perf_counter()-tick
        progress('online_joint_gradient',update=update);tick=time.perf_counter();optimizer.zero_grad()
        accumulator=ShardedGradientAccumulator(model)
        losses=actor.backward(c,y,z,scores['rewards'],format=fmt,accumulator=accumulator,world=world)
        del accumulator;norm=optimizer.finish_gradients(scale=256);training=time.perf_counter()-tick
        tick=time.perf_counter();optimizer.step();sharded_ema(scorer,model,decay=.99);drain_runtime(actor.device)
        optimizer_time=time.perf_counter()-tick
        row=dict(update=update,optimizer_counter=optimizer.update,sampling_seconds=sampling,scoring_seconds=scoring,
            gradient_seconds=training,optimizer_ema_seconds=optimizer_time,total_update_seconds=sampling+scoring+training+optimizer_time,
            absolute_mean_gain=float(scores['absolute_gain'].mean()),reward_sd=float(scores['absolute_gain'].std()),
            gradient_norm=norm,weighted_losses=losses)
        records.append(row)
        if rank==0:(out/'updates.json').write_text(json.dumps(records,indent=2)+'\n')
        if not all(torch.isfinite(torch.tensor(x)) for x in [row['absolute_mean_gain'],norm,*losses.values()]):raise FloatingPointError('Nonfinite online qualification')
    optimizer.zero_grad();progress('mature_zero_update_gradient_audit')
    view=data.at(plan['mature_audit_index']);c,y=view['prefix'],view['continuation'][:,:4]
    z=actor.sample(c,format=fmt,length=8,group_size=4,generator=generator)
    shared=z.to('neuron',dtype=torch.float32);dist.broadcast(shared,src=0);z=shared.cpu().long()
    scores=reader.score(c,y,z,format=fmt);shared=scores['rewards'].to('neuron');dist.broadcast(shared,src=0);reward=shared.cpu()
    accumulator=ShardedGradientAccumulator(model);mature_losses=actor.backward(c,y,z,reward,format=fmt,accumulator=accumulator,world=world)
    del accumulator;mature_norm=optimizer.finish_gradients(scale=256,clip=1e30)
    torch.save({n:local_parameter(p.grad).detach().cpu() for n,p in model.named_parameters()},out/f'rank{rank}-mature-gradients.pt')
    if rank==0:torch.save(dict(prefix=c,targets=y,workspace=z,rewards=reward),out/'private-mature-inputs.pt')
    optimizer.zero_grad();drain_runtime(actor.device);progress('streamed_final_full_state')
    receipt=save_complete_state(out/f'rank{rank}-u101.zip',model,scorer,optimizer,generator,
        extra=dict(plan_sha256=a.plan_sha,data_cursor=101*world,repeated_single_train_anchor=False,source_batches_sha256=plan['data_report_sha256']))
    dist.barrier()
    if rank==0:
        warm=[r['total_update_seconds'] for r in records[1:]]
        result=dict(status='PASS_WORKSPACE_ONLINE_THROUGHPUT_EXECUTION_PENDING_MATURE_AUDIT',world=world,
            wall_seconds=time.time()-started,complete_fresh_process_parent_restore_exact=False,intentional_new_optimizer_from_common_c0=True,
            uninterrupted_next_update_bit_exactness_established=False,actual_qualification_policy_optimizer_updates=101,
            starting_optimizer_counter=0,final_optimizer_counter=optimizer.update,scientific_policy_fit_updates=0,
            cold_update_seconds=records[0]['total_update_seconds'],warm_updates=100,warm_update_mean_seconds=sum(warm)/len(warm),
            warm_update_median_seconds=sorted(warm)[len(warm)//2],unique_source_groups=plan['qualification_source_groups'],unique_source_positions=plan['qualification_unique_source_targets'],
            ordinary_label_exposures=101*world*(c.numel()-1+y.numel()),reader_label_exposures=101*world*4*y.numel(),
            sampled_tokens=102*world*4*8,actor_ledger=actor.ledger(),scorer_ledger=reader.ledger(),
            mature_losses=mature_losses,mature_gradient_norm=mature_norm,final_checkpoint_bytes_per_rank=receipt['bytes'],
            fallback_ops=torch_neuronx.get_fallback_ops(),policy_fit_admitted=False,
            scope='One cold plus100warm actual online generate/score/gradient/AdamW/EMA updates on 6464 admitted TRAIN source views across64 ranks. No novel-data scientific fit or learned-thinking claim. All loss arithmetic is stable FP32;batched G4 sampling. Final full-state and complete unclipped gradient on a fixed unseen TRAIN group saved for independent mature CPU audit before scientific admission.')
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);run(p.parse_args())
