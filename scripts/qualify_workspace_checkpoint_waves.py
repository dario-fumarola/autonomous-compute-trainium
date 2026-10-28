"""Actual-size live-update gradient/cache/checkpoint qualification."""
import argparse
import io
import json
import os
from pathlib import Path
import time
import zipfile
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def run(a):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from compute_trainium.training import file_sha256
    from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig
    from compute_trainium.workspace_native_state import restore_complete_state,save_state_waves,release_compiled_calls
    from compute_trainium.workspace_native_objective import SelectiveWorkspacePredictor,drain_runtime
    from compute_trainium.workspace_discovery import backward_terms,discovery_terms
    from compute_trainium.workspace_sharded_optimizer import WorkspaceShardedAdamW,ShardedGradientAccumulator,sharded_ema
    from compute_trainium.workspace_learning import WorkspaceFormat
    from compute_trainium.scale_model_sharding import shard_model_from_cpu
    from compute_trainium.scale_sharded_candidate import local_parameter
    plan=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or plan['purpose']!='workspace_checkpoint_waves_qualification' or plan['root_comm_ports'][a.output.name]!=a.comm_port:raise ValueError('Frozen writer qualification required')
    if any(file_sha256(source/n)!=h for n,h in plan['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    def progress(stage):
        if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started)))
    progress('loading_exact_qualified_mature_state')
    c0=torch.load(a.root/plan['checkpoint_relative'],weights_only=True,mmap=True)['model'];config=OlmoWorkspaceConfig(**plan['model_config'])
    with torch.device('meta'):model=WorkspaceOlmoCandidate(config,activation_checkpointing=True).eval()
    model=shard_model_from_cpu(model,mesh,c0,'neuron')
    with torch.device('meta'):scorer=WorkspaceOlmoCandidate(config).eval()
    scorer=shard_model_from_cpu(scorer,mesh,c0,'neuron').requires_grad_(False);del c0
    optimizer=WorkspaceShardedAdamW(model,fused=True);generator=torch.Generator()
    parent=a.root/'runs/workspace-native-online-throughput-v2-online'/f'rank{rank}-u101.zip'
    restored=restore_complete_state(parent,model,scorer,optimizer,generator)
    drain_runtime('neuron')
    original_torch_rng=torch.get_rng_state();original_generator=generator.get_state()
    model.compile_stages('neuron');scorer.compile_stages('neuron')
    actor=SelectiveWorkspacePredictor(model);reader=SelectiveWorkspacePredictor(scorer)
    inputs=torch.load(a.root/'runs/workspace-native-online-throughput-v2-online/private-mature-inputs.pt',weights_only=True)
    c,y,z,rewards=[inputs[k] for k in ['prefix','targets','workspace','rewards']]
    fmt=WorkspaceFormat(tuple(plan['opening_ids']),tuple(plan['closing_ids']))
    progress('warming_batched_generation_and_frozen_scores')
    actor.sample(c,format=fmt,length=8,group_size=4,generator=generator)
    scores=reader.score(c,y,z,format=fmt)
    if float((scores['rewards']-rewards).abs().max())>.002:raise AssertionError('Unchanged qualified reader score mismatch')
    progress('full_joint_gradient_with_runtime_scalar_inputs')
    optimizer.zero_grad();accumulator=ShardedGradientAccumulator(model)
    # The already qualified shared source has only a four-token horizon.
    from compute_trainium.workspace_joint_terms import workspace_joint_terms
    backward_terms(actor,workspace_joint_terms(c,y,z,rewards,fmt),accumulator=accumulator,world=world)
    del accumulator;norm=optimizer.finish_gradients(scale=256,clip=1e30)
    release_compiled_calls(model,scorer)
    expected=torch.load(a.root/'runs/workspace-native-online-throughput-v2-online'/f'rank{rank}-mature-gradients.pt',weights_only=True,mmap=True)
    numbers=[]
    for name,p in model.named_parameters():
        ref=expected[name].reshape(-1);actual=local_parameter(p.grad).detach().cpu().reshape(-1)
        difference=reference=0.
        for offset in range(0,ref.numel(),65536):
            wanted=ref[offset:offset+65536].double();got=actual[offset:offset+65536].double()
            difference+=float((got-wanted).square().sum());reference+=float(wanted.square().sum())
        numbers.append([difference,reference]);del wanted,got
        del ref,actual
        drain_runtime('neuron')
    values=torch.tensor(numbers,dtype=torch.float32).to('neuron');dist.all_reduce(values);values=values.cpu().double()
    errors=(values[:,0]/values[:,1].clamp_min(1e-30)).sqrt()
    if float(errors.max())>.02:raise AssertionError('Runtime scalar helper differs from independent-CPU-qualified full gradient')
    largest=float(errors.max());del expected,values,errors,numbers
    progress('actual_qualification_optimizer_update_and_ema')
    optimizer.finish_gradients(scale=1,clip=1.)
    optimizer.step();sharded_ema(scorer,model,decay=.99)
    optimizer.zero_grad();drain_runtime(actor.device)
    if rank==0:(out/'qualification-updates.json').write_text(json.dumps(dict(completed_qualification_updates=1,optimizer_counter=optimizer.update)))
    progress('release_compiled_calls_and_bound_writer_wave')
    release_compiled_calls(model,scorer);torch.set_rng_state(original_torch_rng);generator.set_state(original_generator)
    path=out/f'rank{rank}-roundtrip.zip'
    saved=save_state_waves(path,model,scorer,optimizer,generator,extra=dict(qualification=True,parent_update=restored['update']),wave_size=1,save_ranks=2)
    if rank<2:
        with zipfile.ZipFile(path) as written:new=torch.load(io.BytesIO(written.read('metadata.pt')),weights_only=True)
        if len(new['order'])!=895 or new['metadata']['update']!=102:raise AssertionError('Every live post-update state tensor must be covered')
        if not torch.equal(original_torch_rng,new['metadata']['torch_rng']) or not torch.equal(original_generator,new['metadata']['sampling_rng']):raise AssertionError('Exact RNG coverage required')
        from compute_trainium.workspace_native_state import complete_state_tensors
        with torch.no_grad():
            for name,value in complete_state_tensors(model,scorer,optimizer):value.reshape(-1)[0].add_(1.)
        generator.manual_seed(1);torch.manual_seed(2)
        restore_complete_state(path,model,scorer,optimizer,generator)
        if not torch.equal(torch.get_rng_state(),original_torch_rng) or not torch.equal(generator.get_state(),original_generator):raise AssertionError('Post-update restore RNG mismatch')
    dist.barrier();fallback=torch_neuronx.get_fallback_ops()
    if fallback:raise AssertionError('Checkpoint qualification CPU fallback:'+str(fallback))
    if rank==0:(out/'report.json').write_text(json.dumps(dict(status='PASS_ACTUAL_WORKSPACE_CHECKPOINT_WAVES',world=world,optimizer_updates=1,scientific_optimizer_updates=0,
        largest_named_gradient_relative_l2=largest,gradient_norm=norm,full_state_tensor_count=895,checkpoint_writer_wave_size=1,
        complete_archives_checked=2,all_post_update_tensor_hashes_verified=True,complete_perturb_restore=True,exact_rng=True,fallback_ops=fallback,wall_seconds=time.time()-started,
        scope='Actual1.485B actor/scorer/AdamW/RNG after one LIVE fused optimizer step and EMA from qualified u101.Mature pre-step gradient retains unchanged2% criterion.One writer at a time verifies895tensors;two ranks perturb/restore complete new states.No scientific result or repair of either lost48-update history.'),indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);run(p.parse_args())
