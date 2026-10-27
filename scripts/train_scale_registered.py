"""Head-free qualified math under an explicit gate/TUNE/development lineage."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]
os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'


def lineage(plan_sha,case):
    return dict(study_sha256=plan_sha,case=case['lineage_case'],phase=case['phase'],
        initializer_sha256=case['initializer_sha256'],packing_fingerprint=case['packing_fingerprint'],
        model_role=case['model_role'],state_seed=case['state_seed'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--case',required=True);p.add_argument('--comm-port',type=int,required=True);a=p.parse_args();started=time.time()
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
    from compute_trainium.scale_model_sharding import shard_model_from_cpu
    from compute_trainium.scale_training_state import ScaleTrainingOptimizer,save_training_shard,load_training_shard
    from compute_trainium.scale_training_schedule import rates,apply_rates
    from compute_trainium.scale_work_ledger import fixed_update_work
    from compute_trainium.scale_sharded_fixture import gather_named
    from compute_trainium.parcae_training_candidate import initial_state,map_after_shared_validation
    from compute_trainium.metrics import real_token_losses
    from compute_trainium.training import PackedCorpus,file_sha256,atomic_checkpoint
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen scale study changed')
    plan=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1];case=plan['cases'][a.case]
    if (plan['purpose']!='scale_registered_development_v1' or plan['phases'][a.output.name]['case']!=a.case
            or plan['root_comm_ports'][a.output.name]!=a.comm_port):raise ValueError('Exact registered phase required')
    if any(file_sha256(source/f)!=sha for f,sha in plan['worker_source_sha256'].items()):raise AssertionError('Frozen worker source changed')
    if time.time()>plan['absolute_deadline_unix']:raise TimeoutError('Original scale deadline')
    if case['phase']!='gate':
        gates=json.loads((a.root/plan['trainer_gate_receipt']).read_text())
        if gates['status']!='PASS_SCALE_REGISTERED_TRAINER_AND_REPLAY':raise ValueError('Trainer gate required before TUNE/main')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=64:raise ValueError('Qualified64-core geometry required')
    a.output.mkdir(exist_ok=True);folder=a.root/case['artifact_directory'];folder.mkdir(parents=True,exist_ok=True)
    packed_path=a.root/plan['data']['packed'];initial_file=a.root/case['initializer']
    if rank==0:
        if file_sha256(initial_file)!=case['initializer_sha256']:raise AssertionError('Fresh initializer changed')
        checked=PackedCorpus.from_mapped(packed_path,1024,23,plan['data']['source_sha256'])
        if checked.fingerprint!=case['packing_fingerprint']:raise AssertionError('Source packing changed')
        del checked
    dist.barrier();packed=map_after_shared_validation(packed_path,plan['data']['manifest_sha256'])
    initial=torch.load(initial_file,map_location='cpu',mmap=True,weights_only=False)
    if (initial['purpose']!='scale_registered_unupdated_initializer' or initial['phase']!=case['phase']
            or initial['seed']!=case['seed'] or initial['optimizer_updates']!=0 or initial['model_forwards']!=0
            or initial['eligible_for_science']!=(case['phase']!='gate')):
        raise ValueError('Only original random initialization for this exact phase is allowed')
    config=ParcaeCandidateConfig(**case['model']);recurrent=case['model_role']=='fixed_two';depth=2 if recurrent else 1
    if initial['config']!=vars(config) or initial['recurrent']!=recurrent:raise ValueError('Initializer/config mismatch')
    torch.manual_seed(case['seed']+rank)
    with torch.device('meta'):model=ParcaePrimitiveSDPACandidate(config,recurrent=recurrent,initialize=False,fp16_body=True)
    mesh=init_device_mesh('neuron',(64,));model=shard_model_from_cpu(model,mesh,initial['model'],'neuron');del initial
    signature=lineage(a.plan_sha,case)
    optimizer=ScaleTrainingOptimizer(model,lineage=signature,lr=case['peak_lr'],muon_lr=case['peak_lr'],fused_adamw=True)
    per_update=fixed_update_work(config,case['model_role'])['total_per_update']
    if per_update!=case['work_per_update']:raise AssertionError('Frozen arbitrary-precision work differs')
    update=0
    if case.get('resume'):
        parent=case['resume'];update=load_training_shard(model,optimizer,a.root/parent['directory'],signature,parent['update'],packed,per_update)
    origin=update;model.compile_stages('neuron')
    def objective(logits,targets,mask):return real_token_losses(logits,targets,mask).sum()
    loss=torch.compile(objective,backend='neuron',fullgraph=True,dynamic=False)
    warm_seconds=checkpoint_seconds=capture_seconds=0.;rows=[]
    while update<case['stop_update']:
        if time.time()>plan['absolute_deadline_unix']:raise TimeoutError('Original absolute scale deadline')
        if shutil.disk_usage(folder).free<35*1024**3:raise RuntimeError('Protected artifact storage reserve')
        cursor=update*512;counts=list(map(int,packed.counts(cursor,512)));target_count=sum(counts)
        exposure=sum(map(int,packed.counts(0,cursor+512)))
        values=rates(exposure,warmup_targets=case['warmup_targets'],cooldown_targets=case['cooldown_targets'],
            end_targets=case['end_targets'],peak_lr=case['peak_lr']);apply_rates(optimizer,values)
        optimizer.zero_grad();nll=torch.zeros((),device='neuron');noise=[]
        dist.barrier();torch.neuron.synchronize();tick=time.perf_counter()
        capture=update in case['capture_before_updates']
        for micro in range(8):
            window=cursor+micro*64+rank;batch=packed.batch([window],0)[0]
            state=initial_state(config,window,case['state_seed']) if recurrent else None
            if capture and recurrent:noise.append(state.clone())
            inputs,targets,mask=[t.contiguous().to('neuron') for t in [batch.inputs,batch.targets,batch.target_mask]]
            logits=model(inputs,state.contiguous().to('neuron') if recurrent else None,total_steps=depth,backward_cap=depth)
            scalar=loss(logits,targets,mask);(scalar*(8192/target_count)).backward();nll.add_(scalar.detach())
        norm=optimizer.finish_gradients(scale=8192);dist.all_reduce(nll);torch.neuron.synchronize();export_elapsed=0.
        if capture:
            export=time.perf_counter();gradients=gather_named(model,gradients=True)
            if recurrent:torch.save(dict(states=torch.cat(noise),windows=[cursor+i*64+rank for i in range(8)]),folder/f'noise-u{update}-rank{rank}.pt')
            if rank==0:atomic_checkpoint(folder/f'gradient-u{update}.pt',dict(gradients=gradients,clipped=True,
                loss=float(nll.cpu())/target_count,raw_gradient_norm=norm,source_cursor=cursor,real_targets=target_count,
                signature=signature,optimizer_rates=values))
            del gradients,noise;dist.barrier();export_elapsed=time.perf_counter()-export;capture_seconds+=export_elapsed
        optimizer.step();torch.neuron.synchronize();elapsed=time.perf_counter()-tick-export_elapsed
        if update>origin:warm_seconds+=elapsed
        update+=1
        if update in case['save_updates']:
            before=time.perf_counter();save_training_shard(model,optimizer,folder,signature,update,packed,per_update)
            checkpoint_seconds+=time.perf_counter()-before
        if rank==0:
            row=dict(update=update,cursor=update*512,real_targets=exposure,domain_targets=list(map(int,packed.counts(0,update*512))),
                work=update*per_update,global_train_nll=float(nll.cpu())/target_count,gradient_norm=norm,
                lr=values['lr'],update_seconds=elapsed,elapsed_seconds=time.time()-started)
            rows.append(row)
            # Keep complete histories on disk without quadratic report growth.
            with (a.output/'updates.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
            (a.output/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
            if update==origin+1 or update%100==0:print(json.dumps(row),flush=True)
    if rank==0:
        result=dict(status='CAPTURED_SCALE_REGISTERED_TRAINING_PHASE',world=64,case=a.case,phase=case['phase'],signature=signature,
            stop_update=update,actual_optimizer_updates=update-origin,real_targets=rows[-1]['real_targets'],work=update*per_update,
            plan_sha256=a.plan_sha,warm_seconds=warm_seconds,checkpoint_seconds=checkpoint_seconds,
            gradient_capture_seconds=capture_seconds,wall_seconds=time.time()-started,
            cooldown_complete=update==case['endpoint_updates'],eligible_for_scientific_comparison=False,
            scope='Numerically captured registered phase only; prescribed independent final audits/readouts remain required.')
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
