"""Finite121-update objective qualification;never a scientific fit launcher."""
import argparse
import contextlib
import gc
import json
import os
import time
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from compute_trainium.depth_mixture import routed_indices
from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig
from compute_trainium.future_span_training import FutureSpanObjective,objective_coefficients,training_window
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer,changing_qualification_schedule
from compute_trainium.parcae_training_candidate import finish_gradients,map_after_shared_validation
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def full_state(model,optimizer):
    return {'model':dict(sorted(model.state_dict().items())),'optimizer':optimizer.state_dict()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--case',required=True);parser.add_argument('--mode',choices=['warm','replay'],required=True)
    parser.add_argument('--comm-port',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path);parser.add_argument('--resume-sha');parser.add_argument('--corpus')
    args=parser.parse_args();started=time.time()
    if file_sha256(args.plan)!=args.plan_sha:raise AssertionError('Frozen future qualification plan changed')
    plan=json.loads(args.plan.read_text())
    if plan['purpose']!='finite_future_span_training_qualification' or not plan['qualification_only']:
        raise ValueError('Explicit bounded qualification-only plan required')
    if plan['root_comm_ports'].get(args.output.name)!=args.comm_port:
        raise ValueError('Exact registered lowercase phase identity/communication port required')
    if (args.mode=='replay')!=(args.resume is not None) or (args.resume is None)!=(args.resume_sha is None):
        raise ValueError('Replay requires an explicit checksum-qualified120-update checkpoint')
    code_root=Path(__file__).resolve().parents[1]
    if any(file_sha256(code_root/p)!=sha for p,sha in plan['worker_source_sha256'].items()):
        raise AssertionError('Frozen future qualification worker source changed')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(args.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=32:raise ValueError('Complete32-rank qualification required')
    args.output.mkdir(exist_ok=True)
    preparation=args.root/plan['preparation'];manifest_file=preparation/'manifest.json'
    if file_sha256(manifest_file)!=plan['preparation_manifest_sha256']:raise AssertionError('CPU preparation changed')
    manifest=json.loads(manifest_file.read_text());case=plan['cases'][args.case]
    if case not in manifest['cases']:raise AssertionError('Unprepared qualification cell')
    model_config=ParcaeCandidateConfig(**manifest['model']);span=FutureSpanConfig(**manifest['future_config'])
    packed_path=args.root/manifest['data']['packed'];initial_file=preparation/manifest['initial_file']
    table_info=manifest['depths'][case['arm']];depth_file=preparation/table_info['file']
    if rank==0:
        for path,sha in [(initial_file,manifest['initial_sha256']),(depth_file,table_info['sha256']),
                         (preparation/manifest['prefix_file'],manifest['prefix_sha256'])]:
            if file_sha256(path)!=sha:raise AssertionError('Qualification initialization/depth/count file changed')
        verified=PackedCorpus.from_mapped(packed_path,1024,23,manifest['data']['corpus_sha256'])
        if verified.fingerprint!=manifest['data']['packing_fingerprint']:raise AssertionError('Original natural packing changed')
        del verified
    dist.barrier()
    packed=map_after_shared_validation(packed_path,manifest['data']['manifest_sha256'])
    depths=np.load(depth_file,mmap_mode='r');anchor_prefix=np.load(preparation/manifest['prefix_file'],mmap_mode='r')
    if depths.shape!=(121,16,32,2) or not np.array_equal(depths[...,0],depths[...,1]):
        raise AssertionError('Fixed full-gradient qualification depth table required')
    saved=torch.load(initial_file,map_location='cpu',mmap=True,weights_only=False)
    if not saved['qualification_only']:raise ValueError('Qualification-only initializer required')
    model=FutureSpanCandidate(model_config,span,head_seed=manifest['head_seed'],initialize=False,fp16_body=True)
    model.load_state_dict(saved['model'],strict=True);del saved
    if digest(dict(sorted(model.state_dict().items())))!=manifest['model_digest']:
        raise AssertionError('Common body/head initialization differs')
    model.to('neuron');optimizer=ParcaeRuntimeScalarOptimizer(model,gate_policy='release_muon',
        lr=1e-4,muon_lr=1e-4,fused_adamw=True,muon_backend=None)
    signature={'plan_sha256':args.plan_sha,'case':case,'preparation_manifest_sha256':plan['preparation_manifest_sha256'],
        'depth_table_sha256':table_info['sha256'],'initial_sha256':manifest['initial_sha256'],
        'state_seed':manifest['state_seed'],'routing_seed':manifest['routing_seed'],
        'precision':'fp16_body_fp32_heads_masters_moments','manual_loss_scale':8192,
        'world':32,'accumulation':16,'gradient_rule':'all_sampled_steps'}
    update=cursor=exposure=anchor_exposure=model_work=optimizer_work=0
    recovered_digest=None;recovered_rng=None
    if args.resume:
        if rank==0 and file_sha256(args.resume)!=args.resume_sha:raise AssertionError('Qualification resume file changed')
        dist.barrier();saved=torch.load(args.resume,map_location='cpu',mmap=True,weights_only=False)
        if (saved['signature']!=signature or saved['qualification_only'] is not True
                or saved['update']!=120 or saved['cursor']!=120*512):
            raise ValueError('Exact same-case120-update qualification recovery required')
        if (saved['real_targets']!=sum(packed.counts(0,120*512))
                or saved['valid_anchors']!=int(anchor_prefix[120*512])):
            raise AssertionError('Recovered source and auxiliary cursors differ')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        recovered_digest=digest(full_state(model,optimizer))
        if recovered_digest!=saved['full_state_digest']:raise AssertionError('Exact optimizer/model restore failed')
        update,cursor,exposure,anchor_exposure=120,120*512,saved['real_targets'],saved['valid_anchors']
        model_work,optimizer_work=saved['model_matmuls'],saved['optimizer_matmuls']
        recovered_rng=saved['cpu_rng_by_rank'][rank].clone();del saved
    origin_update,origin_exposure=update,exposure
    torch._dynamo.config.optimize_ddp=False;model.compile_stages('neuron')
    ddp=DDP(FutureSpanObjective(model,backend='neuron'),device_ids=None,broadcast_buffers=False,
            bucket_cap_mb=64,gradient_as_bucket_view=True)
    if recovered_rng is not None:
        torch.set_rng_state(recovered_rng)
        if not torch.equal(torch.get_rng_state(),recovered_rng):raise AssertionError('Exact per-rank CPU RNG restore failed')
    scale=8192;one_optimizer=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')
    warm_start=warm_end=None;warm_tokens=warm_model=warm_optimizer=0
    checkpoint_sha=checkpoint_digest=None;probe_records=[];progress=[]

    def replica_proof(tag,extra=None):
        state_digest=digest(full_state(model,optimizer))
        record={'full_state_digest':state_digest,'fallback_ops':list(torch_neuronx.get_fallback_ops()),**(extra or {})}
        (args.output/f'{tag}-rank{rank}.json').write_text(json.dumps(record)+'\n');dist.barrier()
        if rank==0:
            rows=[json.loads((args.output/f'{tag}-rank{i}.json').read_text()) for i in range(32)]
            if len({row['full_state_digest'] for row in rows})!=1:raise AssertionError('Complete body/head/optimizer replicas differ')
            if any(set(row['fallback_ops'])-{'aten::_to_copy'} for row in rows):raise AssertionError('Unexpected qualification fallback')
            if extra and 'gradient_digest' in extra and len({row['gradient_digest'] for row in rows})!=1:
                raise AssertionError('Complete global gradients differ across replicas')
            return state_digest,rows
        return state_digest,None

    while update<121:
        if update==20:torch.neuron.synchronize();warm_start=time.time()
        if update==120:
            torch.neuron.synchronize();warm_end=time.time() if args.mode=='warm' else None
            torch.save(torch.get_rng_state(),args.output/f'cpu-rng-rank{rank}.pt');dist.barrier()
            state_digest,_=replica_proof('checkpoint120')
            if rank==0:
                file=args.output/'full-state-u120.pt'
                atomic_checkpoint(file,{**full_state(model,optimizer),'signature':signature,'qualification_only':True,
                    'update':update,'cursor':cursor,'real_targets':exposure,'valid_anchors':anchor_exposure,
                    'model_matmuls':model_work,'optimizer_matmuls':optimizer_work,'full_state_digest':state_digest,
                    'cpu_rng_by_rank':[torch.load(args.output/f'cpu-rng-rank{i}.pt',weights_only=True) for i in range(32)]})
                checkpoint_sha=file_sha256(file);checkpoint_digest=state_digest
            dist.barrier()
        total=sum(packed.counts(cursor,512));anchors=int(anchor_prefix[cursor+512]-anchor_prefix[cursor])
        coefficients=objective_coefficients(total,anchors,case['objective_weight'],world).to('neuron')
        rate=changing_qualification_schedule(optimizer,update);optimizer.zero_grad(set_to_none=True)
        probe=update in [0,120];logged=probe or (update+1)%10==0
        totals=torch.zeros(2,device='neuron') if logged else None
        local_rows=[];noise=[]
        for micro in range(16):
            window=routed_indices(cursor,micro,rank,32,1,16,manifest['routing_seed'])[0]
            values=training_window(packed,window,model_config,span,manifest['state_seed'])
            n,k=map(int,depths[update,micro,rank])
            if probe:
                noise.append(values[3]);local_rows.append({'window':window,'rank':rank,'micro':micro,
                    'n':n,'k':k,'h0_digest':digest(values[3]),'valid_anchors':int(values[-1].sum())})
            transferred=[v.contiguous().to('neuron') for v in values]
            with ddp.no_sync() if micro<15 else contextlib.nullcontext():
                value,parts=ddp(*transferred,coefficients,n,k);(value*scale).backward()
            if logged:totals+=parts
        norm=finish_gradients(model,scale=scale,clip=float('inf') if probe else 1.)
        if logged:dist.all_reduce(totals)
        if probe:
            vectors={name:p.grad.detach().cpu() for name,p in model.named_parameters()}
            if any(not torch.isfinite(g).all() for g in vectors.values()):raise FloatingPointError('Nonfinite complete objective gradient')
            noise_file=args.output/f'h0-u{update}-rank{rank}.npy';np.save(noise_file,torch.cat(noise).numpy())
            _,replicas=replica_proof(f'probe{update}',{'gradient_digest':digest(vectors),'rows':local_rows,
                'noise_sha256':file_sha256(noise_file)})
            if rank==0:
                rows=sorted([row for replica in replicas for row in replica['rows']],key=lambda r:r['window'])
                if [row['window'] for row in rows]!=list(range(cursor,cursor+512)) or sum(row['valid_anchors'] for row in rows)!=anchors:
                    raise AssertionError('Full global source/auxiliary coverage differs')
                vector_file=args.output/f'gradient-u{update}.pt';atomic_checkpoint(vector_file,{'gradients':vectors,
                    'ntp_nll':float(totals[0].cpu())/total,'future_mean':float(totals[1].cpu())/max(1,anchors)})
                probe_records.append({'update':update,'cursor':cursor,'real_targets':total,'valid_anchors':anchors,
                    'rows':rows,'noise_sha256_by_rank':[row['noise_sha256'] for row in replicas],
                    'gradient_file':vector_file.name,'gradient_sha256':file_sha256(vector_file),
                    'raw_gradient_norm':norm})
            del vectors;finish_gradients(model,scale=1,clip=1.)
        optimizer.step()
        if update==0 and rank==0:atomic_checkpoint(args.output/'after-first-update.pt',full_state(model,optimizer))
        work=sum(matmul_ledger(model_config,1024,total_steps=int(n),backward_cap=int(k))['training_matmul_flops']
                 for n,k in depths[update].reshape(-1,2))
        work+=512*6*len(span.positions)*model_config.dim*model_config.vocab_size
        model_work+=work;optimizer_work+=32*one_optimizer
        if 20<=update<120:warm_tokens+=total;warm_model+=work;warm_optimizer+=32*one_optimizer
        update+=1;cursor+=512;exposure+=total;anchor_exposure+=anchors
        if logged:
            row={'update':update,'real_targets':exposure,'valid_anchors':anchor_exposure,
                'global_train_nll':float(totals[0].cpu())/total,'global_future_loss':float(totals[1].cpu())/max(1,anchors),
                'gradient_norm_before_clip':norm,'lr':rate,'elapsed_seconds':time.time()-started}
            if rank==0:progress.append(row);print(json.dumps(row),flush=True)
        if probe:gc.collect()
    final_digest,_=replica_proof('final')
    if rank==0:
        atomic_checkpoint(args.output/'after121.pt',full_state(model,optimizer))
        result={'status':'CAPTURED_FUTURE_SPAN_TRAINING_QUALIFICATION','case':case,'mode':args.mode,
            'signature':signature,'qualification_only':True,'research_optimizer_updates':0,
            'actual_optimizer_calls_per_replica':update-origin_update,'phase_real_targets':exposure-origin_exposure,
            'real_targets':exposure,'valid_anchors':anchor_exposure,'cursor':cursor,'model_matmuls':model_work,
            'optimizer_matmuls_all_replicas':optimizer_work,'model_matmul_scope':'Nominal dense autograd operators including zero-weight control head;native optimizer/dead-code/tile work and communication require separate profiling.',
            'probes':probe_records,'full_checkpoint_sha256':checkpoint_sha,'full_checkpoint_state_digest':checkpoint_digest,
            'first_update_sha256':file_sha256(args.output/'after-first-update.pt') if args.mode=='warm' else None,
            'final_sha256':file_sha256(args.output/'after121.pt'),'final_model_optimizer_digest':final_digest,
            'all_complete_replica_states_equal':True,'recovered_state_digest':recovered_digest,
            'complete_state_on_load_exact':args.mode=='replay','cpu_rng_on_load_exact':args.mode=='replay',
            'warm_updates':100 if args.mode=='warm' else 0,'warm_seconds':warm_end-warm_start if warm_start else None,
            'warm_targets_per_second':warm_tokens/(warm_end-warm_start) if warm_start else None,
            'warm_model_matmuls':warm_model,'warm_optimizer_matmuls_all_replicas':warm_optimizer,
            'progress':progress,'test_accessed':False,'wall_seconds':time.time()-started,
            'scope':'Finite actual-update qualification of four-cell causal objective,full gradients,global token/anchor normalization,FP16 body and FP32 heads/masters,moments,manual8192 unscale,runtime schedule,owned32-rank DDP and exact state load. Independent CPU gradient/first-update/recovery checks are mandatory before any scientific adoption.'}
        (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
