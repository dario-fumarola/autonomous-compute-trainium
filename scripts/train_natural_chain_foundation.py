# Derived from train_natural_chain_pilot.py;only parent selection/purpose/scope differ.
"""A bounded actual mixed NTP/bridge curriculum;preflight stops after update1."""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from compute_trainium.launcher_compat import without_launcher_corpus
from compute_trainium.natural_chain_foundation_state import foundation_initial
from compute_trainium.natural_chain_mixed import MixedConclusionObjective,coefficients,optimizer_schedule,update_matrix_work
from compute_trainium.natural_chain_objective import task_depth
from compute_trainium.natural_chain_pilot_data import source_window,task_row,answer_count
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer
from compute_trainium.parcae_training_candidate import finish_gradients,map_after_shared_validation
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def full_state(model,optimizer):return dict(model=dict(sorted(model.state_dict().items())),optimizer=optimizer.state_dict())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--case',required=True)
    p.add_argument('--mode',choices=['preflight','fit'],required=True);p.add_argument('--comm-port',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--resume-sha');a=p.parse_args(without_launcher_corpus(sys.argv[1:]));started=time.time()
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen mixed pilot plan changed')
    plan=json.loads(a.plan.read_text());case=plan['cases'][a.case];code=Path(__file__).resolve().parents[1]
    if plan['purpose']!='natural_chain_foundation_history_pilot' or plan['root_comm_ports'][a.output.name]!=a.comm_port or any(file_sha256(code/p)!=h for p,h in plan['worker_source_sha256'].items()):raise ValueError('Exact registered mixed pilot required')
    if (a.mode=='fit')!=(a.resume is not None) or (a.resume is None)!=(a.resume_sha is None):raise ValueError('Fit requires the verified complete integration state')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron');rank=dist.get_rank();world=dist.get_world_size()
    if world!=32:raise ValueError('Complete32-rank objective required')
    a.output.mkdir(exist_ok=True);base=a.root/plan['preparation'];initial_info=foundation_initial(plan,case);config=ParcaeCandidateConfig(**plan['model']);data=plan['data']
    if rank==0:
        for f,h in [(base/initial_info['file'],initial_info['sha256']),(base/'tasks.pt',plan['tasks_sha256'])]:
            if file_sha256(f)!=h:raise AssertionError('Mixed pilot body/task asset changed')
        verified=PackedCorpus.from_mapped(a.root/data['packed'],1024,23,data['corpus_sha256'])
        if verified.fingerprint!=data['packing_fingerprint']:raise AssertionError('Original NTP stream changed')
        del verified
    dist.barrier();packed=map_after_shared_validation(a.root/data['packed'],data['manifest_sha256']);tasks=torch.load(base/'tasks.pt',map_location='cpu',mmap=True,weights_only=False)
    model=ParcaePrimitiveSDPACandidate(config,recurrent=True,initialize=False,fp16_body=True);initial=torch.load(base/initial_info['file'],map_location='cpu',mmap=True,weights_only=False)
    if initial['qualification_only'] or initial['purpose']!='natural_chain_foundation_initializer' or initial['role']!=case['arm'] or initial['parent_checkpoint_sha256']!=initial_info['parent_checkpoint_sha256']:raise ValueError('No fitted qualification state may enter science')
    model.load_state_dict(initial['model'],strict=True);del initial
    if digest(dict(sorted(model.state_dict().items())))!=initial_info['model_digest']:raise AssertionError('Declared scientific foundation body differs')
    model.to('neuron');optimizer=ParcaeRuntimeScalarOptimizer(model,gate_policy='release_muon',lr=.001,muon_lr=.001,fused_adamw=True)
    signature=dict(plan_sha256=a.plan_sha,arm=case['arm'],task_weight=case['task_weight'],initial_sha256=initial_info['sha256'],tasks_sha256=plan['tasks_sha256'],packing_fingerprint=data['packing_fingerprint'],world=32,global_ntp=128,global_tasks=128,noise_seed=plan['noise_seed'])
    update=0;ntp_exposure=task_exposure=ntp_work=0;model_work=optimizer_work=0;recovered=None;rng=None
    if a.resume:
        if rank==0 and file_sha256(a.resume)!=a.resume_sha:raise AssertionError('Mixed pilot integration checkpoint changed')
        dist.barrier();saved=torch.load(a.resume,map_location='cpu',mmap=True,weights_only=False)
        if saved['signature']!=signature or saved['qualification_only'] or saved['update']!=1:raise ValueError('Verified complete same-recipe first update required')
        if saved['case']!=a.case and saved['case']!=case.get('parent_case'):raise ValueError('Unregistered mixed pilot branch')
        if saved['case']!=a.case and not saved['stable_trunk']:raise ValueError('No annealed parent extension')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);recovered=digest(full_state(model,optimizer))
        if recovered!=saved['state_digest']:raise AssertionError('Complete mixed state restore failed')
        update=saved['update'];ntp_exposure=saved['ntp_exposure'];task_exposure=saved['task_exposure'];model_work=saved['model_work'];optimizer_work=saved['optimizer_work'];rng=saved['rng_by_rank'][rank].clone();del saved
    origin_update=update;origin_ntp=ntp_exposure;origin_task=task_exposure;torch._dynamo.config.optimize_ddp=False;model.compile_stages('neuron')
    ddp=DDP(MixedConclusionObjective(model,backend='neuron'),device_ids=None,broadcast_buffers=False,bucket_cap_mb=64,gradient_as_bucket_view=True)
    if rng is not None:torch.set_rng_state(rng)
    restored_rng=rng is not None and torch.equal(torch.get_rng_state(),rng)
    optimizer_ops=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon');limit=1 if a.mode=='preflight' else case['total_updates'];progress=[];probe=None;warm_start=warm_end=None;warm_ntp=0
    def proof(tag):
        value=digest(full_state(model,optimizer));(a.output/f'{tag}-rank{rank}.json').write_text(json.dumps(dict(digest=value,fallback_ops=list(torch_neuronx.get_fallback_ops())))+'\n');dist.barrier()
        if rank==0:
            rows=[json.loads((a.output/f'{tag}-rank{i}.json').read_text()) for i in range(32)]
            if len({r['digest'] for r in rows})!=1 or any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in rows):raise AssertionError('Mixed pilot replica/fallback failure')
        return value
    while update<limit:
        if update==20:torch.neuron.synchronize();warm_start=time.time()
        if update==120:torch.neuron.synchronize();warm_end=time.time()
        cursor=plan['ntp_start_cursor']+update*128;ntp_count=sum(packed.counts(cursor,128));task_count=answer_count(tasks,update,case) if case['task_weight'] else 0;n=task_depth(case['arm'],update)
        if cursor+128>packed.windows:raise ValueError('No NTP corpus wrapping')
        optimizer.zero_grad(set_to_none=True);rate=optimizer_schedule(optimizer,update,case['total_updates'],peak=case['peak_lr']);logged=update==0 or (update+1)%10==0 or update+1==limit
        totals=torch.zeros(3,device='neuron') if logged else None;noise_ntp=[];noise_task=[]
        branches=['ntp']+(['task'] if case['task_weight'] else [])
        for kind in branches:
            coeff=coefficients(ntp_count,max(1,task_count),kind=kind,task_weight=case['task_weight']).to('neuron')
            for micro in range(4):
                slot=micro*32+rank
                if kind=='ntp':values=source_window(packed,cursor+slot,config,plan['noise_seed'])
                else:values,_=task_row(tasks,update,slot,case,config,plan['noise_seed'])
                if update==0:(noise_ntp if kind=='ntp' else noise_task).append(values[-1])
                transferred=[v.contiguous().to('neuron') for v in values];last=kind==branches[-1] and micro==3
                with contextlib.nullcontext() if last else ddp.no_sync():
                    loss,parts=ddp(*transferred,coeff,n);(loss*8192).backward()
                if logged:
                    if kind=='ntp':totals[0]+=parts[0]
                    else:totals[1:]+=parts
        norm=finish_gradients(model,scale=8192,clip=float('inf') if update==0 else 1.)
        if logged:dist.all_reduce(totals)
        if update==0:
            vectors={name:p.grad.detach().cpu() for name,p in model.named_parameters()};gd=digest(vectors)
            (a.output/f'gradient-rank{rank}.json').write_text(json.dumps(dict(digest=gd))+'\n')
            np.save(a.output/f'h0-ntp-rank{rank}.npy',torch.cat(noise_ntp).numpy())
            if noise_task:np.save(a.output/f'h0-task-rank{rank}.npy',torch.cat(noise_task).numpy())
            dist.barrier()
            if rank==0:
                if len({json.loads((a.output/f'gradient-rank{i}.json').read_text())['digest'] for i in range(32)})!=1:raise AssertionError('Mixed global gradients differ')
                atomic_checkpoint(a.output/'gradient-u0.pt',dict(gradients=vectors,parts=totals.cpu(),ntp_count=ntp_count,answer_count=task_count,n=n))
                probe=dict(ntp_cursor=cursor,ntp_count=ntp_count,answer_count=task_count,n=n,raw_gradient_norm=norm,
                    gradient_sha256=file_sha256(a.output/'gradient-u0.pt'),noise_ntp_sha256=[file_sha256(a.output/f'h0-ntp-rank{i}.npy') for i in range(32)],
                    noise_task_sha256=[file_sha256(a.output/f'h0-task-rank{i}.npy') for i in range(32)] if noise_task else [])
            del vectors;finish_gradients(model,scale=1,clip=1.)
        optimizer.step();update+=1;ntp_exposure+=ntp_count;task_exposure+=task_count
        whole=update_matrix_work(config,n,task_weight=case['task_weight'],optimizer_per_replica=optimizer_ops);optimizer_work+=32*optimizer_ops;model_work+=whole-32*optimizer_ops
        if 20<update<=120:warm_ntp+=ntp_count
        if logged and rank==0:
            row=dict(update=update,n=n,ntp_exposure=ntp_exposure,answer_token_exposures=task_exposure,ntp_nll=float(totals[0].cpu())/ntp_count,answer_nll=float(totals[1].cpu())/max(1,task_count),eos_nll=float(totals[2].cpu())/128,gradient_norm_before_clip=norm,lr=rate,elapsed_seconds=time.time()-started);progress.append(row);print(json.dumps(row),flush=True)
    torch.neuron.synchronize();state_digest=proof('final');torch.save(torch.get_rng_state(),a.output/f'rng-rank{rank}.pt');dist.barrier()
    if rank==0:
        checkpoint=a.output/f'checkpoint-u{update}.pt';atomic_checkpoint(checkpoint,{**full_state(model,optimizer),'signature':signature,'case':a.case,'purpose':plan['purpose'],'qualification_only':False,'update':update,'ntp_exposure':ntp_exposure,'task_exposure':task_exposure,'model_work':model_work,'optimizer_work':optimizer_work,'stable_trunk':update<=case['total_updates']//2,'state_digest':state_digest,'rng_by_rank':[torch.load(a.output/f'rng-rank{i}.pt',weights_only=True) for i in range(32)]})
        result=dict(status='CAPTURED_MIXED_PILOT_INTEGRATION' if a.mode=='preflight' else 'COMPLETE_MIXED_CURRICULUM_PILOT_FIT',case=a.case,arm=case['arm'],task_weight=case['task_weight'],mode=a.mode,plan_sha256=a.plan_sha,update=update,
            checkpoint_file=checkpoint.name,checkpoint_sha256=file_sha256(checkpoint),full_state_digest=state_digest,all_replicas_equal=True,
            full_state_on_load_exact=recovered is not None,recovered_state_digest=recovered,cpu_rng_on_load_exact=restored_rng,
            actual_optimizer_calls_per_replica=update-origin_update,research_optimizer_updates=update-origin_update,ntp_exposure=ntp_exposure,answer_token_exposures=task_exposure,segment_ntp_targets=ntp_exposure-origin_ntp,segment_answer_targets=task_exposure-origin_task,
            model_matrix_ops=model_work,optimizer_matrix_ops_all_replicas=optimizer_work,probe=probe,progress=progress,warm_updates=100 if warm_end else 0,warm_seconds=warm_end-warm_start if warm_end else None,warm_ntp_targets=warm_ntp,wall_seconds=time.time()-started,test_accessed=False,
            scope='Prospective bounded continuation pilot. First actual update belongs to this registered pilot and counts even on failure. Further updates require independent mixed-global-gradient/optimizer integration pass. Natural NTP remains present;ordered answers and EOS are separately normalized/counted. Each role retains its own qualified NTP foundation,with the same original random initializer and fresh adaptation optimizers. Historical source exposures differ and are fully charged;no new from-scratch confirmation. Native nominal matrices are not chip-time equivalence.')
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
