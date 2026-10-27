"""Actual-shape packed-data warm/replay capture, not scientific training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]
os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--corpus');parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--mode',choices=['warm','replay'],required=True);parser.add_argument('--comm-port',type=int,required=True)
    args=parser.parse_args();started=time.time()
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
    from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
    from compute_trainium.scale_model_sharding import shard_model_from_cpu,StreamingOwnerOptimizer
    from compute_trainium.scale_model_state import save_shard,load_shard,share_exact_replay_shard
    from compute_trainium.scale_sharded_fixture import gather_named
    from compute_trainium.parcae_training_candidate import initial_state,map_after_shared_validation
    from compute_trainium.parcae_runtime_scalars import changing_qualification_schedule
    from compute_trainium.parcae_work_accounting import muon_update_matmuls
    from compute_trainium.metrics import real_token_losses
    from compute_trainium.training import PackedCorpus,file_sha256,atomic_checkpoint
    if file_sha256(args.plan)!=args.plan_sha:raise AssertionError('Exact actual-shape plan required')
    plan=json.loads(args.plan.read_text());code=Path(__file__).resolve().parents[1]
    if (plan['purpose']!='scale_actual_shape_numerical_only' or plan['phases'][args.output.name]['mode']!=args.mode
            or plan['root_comm_ports'][args.output.name]!=args.comm_port):raise ValueError('Exact numerical role required')
    if any(file_sha256(code/f)!=sha for f,sha in plan['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(args.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=64:raise ValueError('64-core actual-shape geometry required')
    args.output.mkdir(exist_ok=True)
    artifact=Path(plan.get('container_artifact_roots',{}).get(args.mode,plan['container_artifact_root']));folder=artifact/args.mode
    folder.mkdir(exist_ok=True)
    initializer=Path(plan['container_initializer']);packing=args.root/plan['data']['packed']
    if rank==0:
        if file_sha256(initializer)!=plan['initializer_sha256']:raise AssertionError('Initializer changed')
        verified=PackedCorpus.from_mapped(packing,1024,23,plan['data']['source_sha256'])
        if verified.fingerprint!=plan['data']['fingerprint']:raise AssertionError('Packing identity changed')
        del verified
    dist.barrier();packed=map_after_shared_validation(packing,plan['data']['manifest_sha256'])
    saved=torch.load(initializer,map_location='cpu',mmap=True,weights_only=False)
    if not saved['qualification_only'] or saved['eligible_for_science'] is not False:
        raise ValueError('Quarantined random numerical initializer required')
    config=ParcaeCandidateConfig(**plan['model']);role=plan.get('model_role','fixed_two')
    if role not in ('fixed_two','ordinary8'):raise ValueError('Explicit qualified scale model role required')
    recurrent=role=='fixed_two';depth=2 if recurrent else 1
    if saved['config']!=vars(config) or saved['recurrent']!=recurrent:raise ValueError('Exact scale config required')
    with torch.device('meta'):model=ParcaePrimitiveSDPACandidate(config,recurrent=recurrent,initialize=False,fp16_body=True)
    mesh=init_device_mesh('neuron',(64,));model=shard_model_from_cpu(model,mesh,saved['model'],'neuron');del saved
    optimizer=StreamingOwnerOptimizer(model,lr=1e-4,muon_lr=1e-4,fused_adamw=True)
    if model.transformer.wte._parameters['weight'] is not model.lm_head._parameters['weight']:
        raise AssertionError('Tied readout alias lost')
    manual_scale=plan.get('manual_loss_scale',8192)
    if manual_scale not in (8192,65536):raise ValueError('Explicit bounded loss scaler required')
    signature=dict(plan_sha256=args.plan_sha,world=64,accumulation=8,global_windows=512,context=1024,
        precision='fp16_body_fp32_primitive_attention_and_head',full_gradient_steps=depth,manual_loss_scale=manual_scale,
        initializer_sha256=plan['initializer_sha256'],packing_fingerprint=packed.fingerprint,
        optimizer_execution='streaming_owner_complete_matrix_fp32_muon_local_fused_adamw',state_seed=2009 if recurrent else None,
        model_role=role)
    update=cursor=exposure=neural_ops=optimizer_ops=0
    if args.mode=='replay':
        warm_artifact=Path(plan.get('container_artifact_roots',{}).get('warm',str(artifact)))
        restored=load_shard(model,optimizer,warm_artifact/'warm',signature,100,packed)
        update,cursor,exposure,neural_ops,optimizer_ops=[restored[k] for k in ['update','cursor','real_targets','neural_ops','optimizer_ops']]
    model.compile_stages('neuron')
    def objective(logits,targets,mask):return real_token_losses(logits,targets,mask).sum()
    loss=torch.compile(objective,backend='neuron',fullgraph=True,dynamic=False)
    warm_seconds=0.;checkpoint_seconds=0.;capture_seconds=0.;rows=[];origin=update
    one_neural=matmul_ledger(config,1024,512,recurrent=recurrent,total_steps=depth,backward_cap=depth)['training_matmul_flops']
    one_optimizer=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')
    while update<101:
        if shutil.disk_usage(artifact).free<35*1024**3:raise RuntimeError('Actual artifact protected reserve')
        target_count=sum(packed.counts(cursor,512));optimizer.zero_grad();noise=[];nll=torch.zeros((),device='neuron')
        capture=update in [0,100] and (args.mode=='warm' or plan.get('replay_capture_gradients',True))
        dist.barrier();torch.neuron.synchronize();tick=time.perf_counter()
        rate=changing_qualification_schedule(optimizer,update)
        for micro in range(8):
            window=cursor+micro*64+rank;batch=packed.batch([window],0)[0]
            state=initial_state(config,window,2009) if recurrent else None
            if recurrent and capture and (update!=0 or plan.get('capture_initial_noise',True)):noise.append(state.clone())
            values=[t.contiguous().to('neuron') for t in [batch.inputs,batch.targets,batch.target_mask]]
            state=state.contiguous().to('neuron') if recurrent else None
            logits=model(values[0],state,total_steps=depth,backward_cap=depth)
            scalar=loss(logits,values[1],values[2]);(scalar*(manual_scale/target_count)).backward();nll.add_(scalar.detach())
        norm=optimizer.finish_gradients(scale=manual_scale)
        torch.neuron.synchronize();before_export=time.perf_counter();export_elapsed=0.
        if capture:
            gradients=gather_named(model,gradients=True)
            if recurrent and noise:
                torch.save(dict(states=torch.cat(noise),windows=[cursor+i*64+rank for i in range(8)]),folder/f'noise-u{update}-rank{rank}.pt')
            dist.all_reduce(nll)
            if rank==0:atomic_checkpoint(folder/f'gradient-u{update}.pt',dict(gradients=gradients,clipped=True,
                loss=float(nll.cpu())/target_count,raw_gradient_norm=norm,source_cursor=cursor,real_targets=target_count))
            del gradients,noise
            dist.barrier();export_elapsed=time.perf_counter()-before_export;capture_seconds+=export_elapsed
        else:dist.all_reduce(nll)
        optimizer.step();torch.neuron.synchronize();elapsed=time.perf_counter()-tick-export_elapsed
        if update>=1:warm_seconds+=elapsed
        update+=1;cursor+=512;exposure+=target_count;neural_ops+=one_neural;optimizer_ops+=one_optimizer
        if update in [1,100,101]:
            begin=time.perf_counter()
            if args.mode=='replay' and plan.get('share_exact_replay',False):
                share_exact_replay_shard(model,optimizer,folder,warm_artifact/'warm',signature,update,packed,neural_ops,optimizer_ops)
            else:save_shard(model,optimizer,folder,signature,update,packed,neural_ops,optimizer_ops)
            checkpoint_seconds+=time.perf_counter()-begin
        if rank==0:
            row=dict(update=update,cursor=cursor,real_targets=exposure,global_train_nll=float(nll.cpu())/target_count,
                gradient_norm=norm,lr=rate,update_seconds=elapsed,elapsed_seconds=time.time()-started)
            rows.append(row);(args.output/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
            if update==1 or update%10==0:print(json.dumps(row),flush=True)
    if rank==0:
        row=dict(status='CAPTURED_SCALE_ACTUAL_SHAPE_NUMERICAL',plan_sha256=args.plan_sha,mode=args.mode,
            qualification_only=True,eligible_for_science=False,parameters=plan['parameters'],world=64,model_role=role,
            actual_optimizer_updates=101-origin,warm_updates=100 if args.mode=='warm' else 0,
            warm_seconds=warm_seconds if args.mode=='warm' else None,checkpoint_seconds=checkpoint_seconds,
            gradient_capture_seconds=capture_seconds,wall_seconds=time.time()-started,
            real_targets=exposure,neural_ops=neural_ops,optimizer_ops=optimizer_ops,progress=rows,
            scientific_fit_admitted=False,scope='Actual-shape numerical/throughput capture only; independent CPU and replay comparison remain required.')
        (args.output/'report.json').write_text(json.dumps(row,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
