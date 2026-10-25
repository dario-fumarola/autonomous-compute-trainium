"""Finite native qualification of original-conclusion loss;not a research fit."""
import argparse
import json
import os
import time
from pathlib import Path
os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from compute_trainium.natural_chain_objective import NaturalChainObjective,task_coefficients,task_depth
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer,changing_qualification_schedule
from compute_trainium.parcae_training_candidate import finish_gradients
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.training import atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def full_state(model,optimizer):return {'model':dict(sorted(model.state_dict().items())),'optimizer':optimizer.state_dict()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--case',choices=['one','recurrent'],required=True)
    p.add_argument('--mode',choices=['warm','replay'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--comm-port',type=int,required=True);p.add_argument('--resume',type=Path);p.add_argument('--resume-sha');a=p.parse_args();started=time.time()
    if file_sha256(a.plan)!=a.plan_sha:raise ValueError('Frozen task plan changed')
    plan=json.loads(a.plan.read_text());code=Path(__file__).resolve().parents[1]
    if (plan['purpose']!='qualification_original_conclusion_driver' or not plan['qualification_only']
        or plan['root_comm_ports'][a.output.name]!=a.comm_port
        or any(file_sha256(code/f)!=h for f,h in plan['worker_source_sha256'].items())):raise ValueError('Exact bounded task qualification required')
    if (a.mode=='replay')!=(a.resume is not None) or (a.resume is None)!=(a.resume_sha is None):raise ValueError('Explicit exact replay checkpoint required')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron');rank=dist.get_rank();world=dist.get_world_size()
    if world!=32:raise ValueError('32 ranks required')
    a.output.mkdir(exist_ok=True);base=a.root/plan['identity'];config=ParcaeCandidateConfig(**plan['model'])
    if rank==0:
        for f,h in [('initial-body.pt',plan['initial_sha256']),('tasks.pt',plan['tasks_sha256'])]:
            if file_sha256(base/f)!=h:raise AssertionError('Frozen task/body asset changed')
    dist.barrier();tasks=torch.load(base/'tasks.pt',map_location='cpu',mmap=True,weights_only=False)
    initial=torch.load(base/'initial-body.pt',map_location='cpu',mmap=True,weights_only=False)
    model=ParcaePrimitiveSDPACandidate(config,recurrent=True,initialize=False,fp16_body=True);model.load_state_dict(initial['model'],strict=True)
    if digest(dict(sorted(model.state_dict().items())))!=plan['model_digest']:raise AssertionError('Common original body differs')
    del initial;model.to('neuron');optimizer=ParcaeRuntimeScalarOptimizer(model,gate_policy='release_muon',lr=1e-4,muon_lr=1e-4,fused_adamw=True)
    signature=dict(plan_sha256=a.plan_sha,arm=a.case,context=128,global_examples=32,source_views_sha256=plan['views_sha256'],initial_sha256=plan['initial_sha256'])
    update=0;recovered=None;rng=None
    if a.resume:
        if rank==0 and file_sha256(a.resume)!=a.resume_sha:raise AssertionError('Replay checkpoint changed')
        dist.barrier();saved=torch.load(a.resume,map_location='cpu',mmap=True,weights_only=False)
        if saved['signature']!=signature or saved['update']!=120:raise ValueError('Same-case exact120-update state required')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);recovered=digest(full_state(model,optimizer))
        if recovered!=saved['state_digest']:raise AssertionError('Complete task state restore failed')
        rng=saved['rng_by_rank'][rank].clone();update=120;del saved
    torch._dynamo.config.optimize_ddp=False;model.compile_stages('neuron');ddp=DDP(NaturalChainObjective(model,backend='neuron'),device_ids=None,broadcast_buffers=False,bucket_cap_mb=64,gradient_as_bucket_view=True)
    if rng is not None:torch.set_rng_state(rng)
    restored_rng=rng is not None and torch.equal(torch.get_rng_state(),rng)
    start_update=update;probes=[];progress=[];checkpoint_sha=checkpoint_digest=None;warm_start=warm_end=None;answer_exposures=0;model_work=0
    optimizer_ops=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon');replay_load=recovered
    def proof(tag):
        value=digest(full_state(model,optimizer));(a.output/f'{tag}-rank{rank}.json').write_text(json.dumps(dict(digest=value,fallback_ops=list(torch_neuronx.get_fallback_ops()))))
        dist.barrier()
        if rank==0:
            rows=[json.loads((a.output/f'{tag}-rank{i}.json').read_text()) for i in range(32)]
            if len({x['digest'] for x in rows})!=1 or any(set(x['fallback_ops'])-{'aten::_to_copy'} for x in rows):raise AssertionError('Task replicas/fallbacks differ')
        return value
    while update<121:
        if update==20:torch.neuron.synchronize();warm_start=time.time()
        if update==120:
            torch.neuron.synchronize();warm_end=time.time() if a.mode=='warm' else None
            torch.save(torch.get_rng_state(),a.output/f'rng-rank{rank}.pt');state_digest=proof('checkpoint120')
            if rank==0:
                f=a.output/'full-state-u120.pt';atomic_checkpoint(f,{**full_state(model,optimizer),'signature':signature,'update':120,'state_digest':state_digest,
                    'rng_by_rank':[torch.load(a.output/f'rng-rank{i}.pt',weights_only=True) for i in range(32)]});checkpoint_sha=file_sha256(f);checkpoint_digest=state_digest
            dist.barrier()
        stage=0 if update<40 else 1 if update<80 else 2;n=task_depth(a.case,update)
        values=[tasks[k][stage,rank:rank+1].contiguous().to('neuron') for k in ['inputs','targets','answer_mask','termination_mask']]
        h0=tasks['h0'][rank:rank+1].contiguous().to('neuron');total=int(tasks['answer_mask'][stage].sum())
        coeff=task_coefficients(total,32,world=world,termination_weight=.1).to('neuron')
        lr=changing_qualification_schedule(optimizer,update);optimizer.zero_grad(set_to_none=True)
        loss,parts=ddp(*values,h0,coeff,n);(loss*8192).backward()
        probe=update in [0,120];norm=finish_gradients(model,scale=8192,clip=float('inf') if probe else 1.)
        dist.all_reduce(parts)
        if probe:
            vectors={name:p.grad.detach().cpu() for name,p in model.named_parameters()};gd=digest(vectors)
            (a.output/f'gradient-u{update}-rank{rank}.json').write_text(json.dumps({'digest':gd}));dist.barrier()
            if rank==0:
                if len({json.loads((a.output/f'gradient-u{update}-rank{i}.json').read_text())['digest'] for i in range(32)})!=1:raise AssertionError('Task global gradients differ')
                f=a.output/f'gradient-u{update}.pt';atomic_checkpoint(f,dict(gradients=vectors,parts=parts.cpu(),answer_tokens=total,stage=stage,n=n))
                probes.append(dict(update=update,stage=stage,n=n,answer_tokens=total,gradient_file=f.name,gradient_sha256=file_sha256(f),gradient_digest=gd,raw_gradient_norm=norm))
            del vectors;finish_gradients(model,scale=1,clip=1.)
        optimizer.step()
        if update==0 and rank==0:atomic_checkpoint(a.output/'after-first-update.pt',full_state(model,optimizer))
        answer_exposures+=total;model_work+=32*matmul_ledger(config,128,total_steps=n,backward_cap=n)['training_matmul_flops'];update+=1
        if probe or update%10==0:
            row=dict(update=update,n=n,stage=stage,answer_nll=float(parts[0].cpu())/total,eos_nll=float(parts[1].cpu())/32,gradient_norm_before_clip=norm,lr=lr,elapsed_seconds=time.time()-started)
            if rank==0:progress.append(row);print(json.dumps(row),flush=True)
    final=proof('final')
    if rank==0:
        atomic_checkpoint(a.output/'after121.pt',full_state(model,optimizer))
        result=dict(status='CAPTURED_NATURAL_CHAIN_DRIVER_QUALIFICATION',case=a.case,mode=a.mode,signature=signature,probes=probes,
            full_checkpoint_sha256=checkpoint_sha,full_checkpoint_state_digest=checkpoint_digest,final_model_optimizer_digest=final,
            full_state_on_load_exact=a.mode=='replay',rng_on_load_exact=restored_rng,recovered_state_digest=replay_load,
            first_update_sha256=file_sha256(a.output/'after-first-update.pt') if a.mode=='warm' else None,final_sha256=file_sha256(a.output/'after121.pt'),
            all_complete_replica_states_equal=True,actual_optimizer_calls_per_replica=update-start_update,answer_token_exposures=answer_exposures,
            nominal_model_matrix_ops=model_work,nominal_optimizer_matrix_ops_all_replicas=(update-start_update)*32*optimizer_ops,
            warm_updates=100 if a.mode=='warm' else 0,warm_seconds=warm_end-warm_start if warm_start else None,
            qualification_only=True,research_optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,progress=progress,
            scope='32 fixed original TRAIN tasks,three bridge stages,normal decoder and explicit EOS loss. Initial body is a previously trained one-pass model;optimizer reset. Qualification updates only,not from-scratch or held-out efficacy. CPU gradients/optimizer audits and replay remain mandatory.')
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
