"""Bounded complete Parcae training benchmark, never a scientific fit launcher."""
import argparse
import contextlib
import ctypes
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
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.depth_mixture import routed_indices
from compute_trainium.metrics import real_token_losses
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_fp16_candidate import ParcaeFP16Candidate
from compute_trainium.parcae_optimizer_candidate import ParcaeOptimizerCandidate
from compute_trainium.parcae_stages_candidate import ParcaeStagesCandidate
from compute_trainium.parcae_training_candidate import (
    configure_optimizer, finish_gradients, gradient_comparison, initial_state,
    map_after_shared_validation, model_work, validate_resume,
)
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.training import PackedCorpus, atomic_checkpoint, cpu_tree, file_sha256
from scripts.qualify_native import digest


def loss_sum(logits,targets,mask):
    return real_token_losses(logits,targets,mask).sum()


class Objective(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
        self.loss=torch.compile(loss_sum,backend='neuron',fullgraph=True,dynamic=False)

    def forward(self,inputs,targets,mask,state,n,k):
        return self.loss(self.model(inputs,state,total_steps=n,backward_cap=k),targets,mask)


def complete_state(model,optimizer):
    return {'model':dict(sorted(model.state_dict().items())),'optimizer':optimizer.state_dict()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--packed-corpus',type=Path,required=True)
    parser.add_argument('--preparation-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=['ordinary','fixed1','poisson2'],required=True)
    parser.add_argument('--accumulation',type=int,choices=[8,16],required=True)
    parser.add_argument('--precision',choices=['fp32','fp16'],required=True)
    parser.add_argument('--mode',choices=['warm','replay'],required=True)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    import torch_neuronx
    torch.set_num_threads(2)
    torch.neuron.set_device(0)
    dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=32 or (args.mode=='replay')!=(args.resume is not None):
        raise ValueError('Declared32-rank primary or separate-process replay required')
    args.output.mkdir(exist_ok=True)
    started=time.time()
    plan=json.loads((args.preparation_root/'plan.json').read_text())
    report=json.loads((args.preparation_root/'cpu-report.json').read_text())
    if report['status']!='COMPLETE_PARCAE_TRAINING_QUALIFICATION_PREPARATION':
        raise ValueError('CPU training preparation incomplete')
    case=next(r for r in plan['cases'] if r['arm']==args.arm and r['accumulation']==args.accumulation)
    table_file=args.preparation_root/f'{args.arm}-a{args.accumulation}-depths.npy'
    initial=plan['initial_states'][case['initial_kind']]
    checkpoint=args.preparation_root/initial['file']
    if rank==0:
        if file_sha256(checkpoint)!=initial['sha256'] or file_sha256(table_file)!=case['depth_table_sha256']:
            raise AssertionError('Initialization or computation schedule changed')
        if file_sha256(args.corpus)!=plan['source_corpus_sha256']:
            raise AssertionError('Natural source corpus changed')
        verified=PackedCorpus.from_mapped(args.packed_corpus,1024,23,plan['source_corpus_sha256'])
        if verified.fingerprint!=plan['packing_fingerprint']:
            raise AssertionError('Natural packing changed')
        del verified
    dist.barrier()
    packed=map_after_shared_validation(args.packed_corpus,plan['packed_manifest_sha256'])
    if packed.context!=1024 or packed.seed!=23:
        raise AssertionError('Qualification packing geometry differs')
    depths=np.load(table_file,mmap_mode='r')
    if depths.shape!=(121,args.accumulation,32,2):
        raise AssertionError('Qualification depth schedule shape differs')
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if not saved['qualification_only']:
        raise ValueError('Qualification-only initial state required')
    config=ParcaeCandidateConfig(**saved['config'])
    recurrent=args.arm!='ordinary'
    if saved['recurrent']!=recurrent:raise AssertionError('Prepared architecture identity differs')
    cls=ParcaeFP16Candidate if args.precision=='fp16' else ParcaeStagesCandidate
    torch.manual_seed(1009)
    model=cls(config,recurrent=recurrent,initialize=False)
    model.load_state_dict(saved['model'])
    if digest(dict(sorted(model.state_dict().items())))!=initial['model_digest']:
        raise AssertionError('Shared unperturbed release initialization changed')
    del saved
    model.to('neuron')
    optimizer=ParcaeOptimizerCandidate(model,gate_policy='release_muon',fused_adamw=True,muon_backend='neuron')
    scale=8192 if args.precision=='fp16' else 1
    signature={'preparation_sha256':file_sha256(args.preparation_root/'plan.json'),
        'world':32,'accumulation':args.accumulation,'context':1024,'arm':args.arm,
        'precision':args.precision,'master_dtype':'fp32','manual_loss_scale':scale,
        'depth_table_sha256':case['depth_table_sha256'],'initial_sha256':initial['sha256'],
        'packing_fingerprint':plan['packing_fingerprint'],'state_seed':2009,'routing_seed':271828,
        'clip':1.,'peak_lr':1e-4,'gate_policy':'release_muon'}
    code_root=Path(__file__).resolve().parents[1]
    signature['training_source_sha256']={p:file_sha256(code_root/p) for p in [
        'scripts/train_parcae_qualification.py','compute_trainium/parcae_training_candidate.py',
        'compute_trainium/parcae_candidate.py','compute_trainium/parcae_stages_candidate.py',
        'compute_trainium/parcae_fp16_candidate.py','compute_trainium/parcae_optimizer_candidate.py']}
    cursor=exposure=update=model_matmuls=optimizer_matmuls=0
    recovered_digest=None
    recovered_rng=None
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',mmap=True,weights_only=False)
        validate_resume(saved,signature)
        if saved['update']!=120:
            raise ValueError('Replay must start at the measured120-update checkpoint')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        recovered_digest=digest(complete_state(model,optimizer))
        if recovered_digest!=saved['model_optimizer_digest']:
            raise AssertionError('Complete model/optimizer state changed on load')
        cursor,exposure,update=saved['cursor'],saved['real_targets'],saved['update']
        model_matmuls,optimizer_matmuls=saved['model_matmuls'],saved['optimizer_matmuls']
        expected_model_work=(model_work(config,depths[:120].reshape(-1,2)) if recurrent else
            matmul_ledger(config,1024,120*32*args.accumulation,recurrent=False)['training_matmul_flops'])
        if exposure!=sum(packed.counts(0,cursor)) or model_matmuls!=expected_model_work:
            raise AssertionError('Recovered source exposure or model-work counter differs')
        expected_optimizer=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')*world*120
        if optimizer_matmuls!=expected_optimizer:
            raise AssertionError('Recovered optimizer-work counter differs')
        recovered_rng=saved['cpu_rng_by_rank'][rank].clone()
        del saved
    origin_update,origin_exposure=update,exposure
    gc.collect();ctypes.CDLL(None).malloc_trim(0)
    torch._dynamo.config.optimize_ddp=False
    model.compile_stages('neuron')
    objective=Objective(model)

    def batch(global_update,micro,worker=rank):
        window=routed_indices(global_update*32*args.accumulation,micro,worker,32,1,args.accumulation,271828)[0]
        item=packed.batch([window],0)[0]
        state=initial_state(config,window,2009) if recurrent else None
        n,k=map(int,depths[global_update,micro,worker])
        values=[t.contiguous().to('neuron') for t in (item.inputs,item.targets,item.target_mask)]
        return item,(*values,state.contiguous().to('neuron') if state is not None else None,n,k)

    # Initial CPU/backend agreement uses the same runtime and exact supplied
    # state. The separate trained full-global audit uses pinned upstream code.
    if args.mode=='warm':
        item,values=batch(0,0)
        local=objective(*values)/int(item.target_mask.sum())
        (local*scale).backward()
        if scale!=1:
            for parameter in model.parameters():parameter.grad.div_(scale)
        if rank==0:
            reference=ParcaeStagesCandidate(config,recurrent=recurrent,initialize=False)
            reference.load_state_dict(torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)['model'])
            with sdpa_kernel(SDPBackend.MATH):
                logits=reference(item.inputs,values[3].detach().cpu() if recurrent else None,total_steps=values[4],backward_cap=values[5])
                value=(F.cross_entropy(logits.flatten(0,1),item.targets.flatten(),reduction='none')*item.target_mask.flatten()).sum()/item.target_mask.sum()
                value.backward()
            expected={'gradients':{n:p.grad.detach() for n,p in reference.named_parameters()},'loss':float(value.detach())}
            actual={n:p.grad.detach().cpu() for n,p in model.named_parameters()}
            delta=gradient_comparison(expected['gradients'],actual)
            difference=float(local.detach().cpu())-expected['loss']
            if not delta['passes_two_percent_gate'] or abs(difference)>.003:
                raise AssertionError('Initial complete local gradient/NLL gate failed')
            (args.output/'initial-gradient-check.json').write_text(json.dumps({**delta,'native_minus_cpu_nll':difference,
                'reference':'Checked FP32 port on native Torch CPU runtime, exact supplied hidden state. Independent upstream full-global trained-point audit follows.'},indent=2)+'\n')
            del expected,actual,reference,logits,value
        model.zero_grad(set_to_none=True)
        del local
    ddp=DDP(objective,device_ids=None,broadcast_buffers=False,bucket_cap_mb=64,gradient_as_bucket_view=True)
    if recovered_rng is not None:
        torch.set_rng_state(recovered_rng)
        if not torch.equal(torch.get_rng_state(),recovered_rng):
            raise AssertionError('Per-rank CPU RNG state changed on load')
    one_optimizer=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')*world
    warm_start=None
    warm_tick=None
    warm_tokens=warm_model=warm_optimizer=0
    warm_seconds=None
    warm_finished=None
    rows=[]
    checkpoint_digest=checkpoint_sha=None
    checkpoint_seconds=0.
    point_noise_sha=point_depth_sha=None

    def save_state():
        nonlocal checkpoint_digest,checkpoint_sha,checkpoint_seconds
        begin=time.time()
        torch.save(torch.get_rng_state(),args.output/f'cpu-rng-rank-{rank}.pt')
        dist.barrier()
        if rank==0:
            full=complete_state(model,optimizer)
            checkpoint_digest=digest(full)
            atomic_checkpoint(args.output/'after-120-full-state.pt',{**full,
                'model_optimizer_digest':checkpoint_digest,'signature':signature,'qualification_only':True,
                'cursor':cursor,'real_targets':exposure,'update':update,
                'model_matmuls':model_matmuls,'optimizer_matmuls':optimizer_matmuls,
                'cpu_rng_by_rank':[torch.load(args.output/f'cpu-rng-rank-{i}.pt',weights_only=True) for i in range(world)],
                'framework_native_rng_ops_used':False,'rng_policy':'Explicit CPU generator keyed by absolute source window; frozen CPU depth table.'})
            checkpoint_sha=file_sha256(args.output/'after-120-full-state.pt')
        dist.barrier()
        checkpoint_seconds=time.time()-begin

    for _ in range(121-origin_update):
        if update==20:
            dist.barrier();torch.neuron.synchronize();warm_start=time.time();warm_tick=time.perf_counter()
        if update==120:
            if args.mode=='warm':
                torch.neuron.synchronize();dist.barrier();warm_finished=time.time()
                warm_seconds=time.perf_counter()-warm_tick
                save_state()
                if rank==0:
                    # Exact states used for the next global objective, independent
                    # of host/native RNG implementation differences.
                    noise=(np.lib.format.open_memmap(args.output/'probe-initial-states.npy',mode='w+',dtype=np.float32,
                        shape=(world*args.accumulation,1024,config.dim)) if recurrent else None)
                    point_depths=np.zeros((world*args.accumulation,2),dtype=np.uint16)
                    for micro in range(args.accumulation):
                        for worker in range(world):
                            window=routed_indices(cursor,micro,worker,32,1,args.accumulation,271828)[0]
                            if noise is not None:noise[window-cursor]=initial_state(config,window,2009)[0].numpy()
                            point_depths[window-cursor]=depths[120,micro,worker]
                    if noise is not None:noise.flush()
                    del noise
                    np.save(args.output/'probe-depths.npy',point_depths)
                    point_noise_sha=file_sha256(args.output/'probe-initial-states.npy') if recurrent else None
                    point_depth_sha=file_sha256(args.output/'probe-depths.npy')
                dist.barrier()
        counts=packed.counts(cursor,world*args.accumulation)
        global_count=sum(counts)
        optimizer.zero_grad(set_to_none=True)
        rate=configure_optimizer(optimizer,update)
        total=torch.zeros((),device='neuron')
        for micro in range(args.accumulation):
            _,values=batch(update,micro)
            with ddp.no_sync() if micro<args.accumulation-1 else contextlib.nullcontext():
                value=ddp(*values)
                (value*(world/global_count)*scale).backward()
            total+=value.detach()
        reduced_loss=update in (0,120)
        if reduced_loss:
            raw_norm=finish_gradients(model,scale=scale,clip=float('inf'))
            dist.all_reduce(total)
            if rank==0:
                target='initial-global-native.pt' if update==0 else 'updated-global-native.pt'
                atomic_checkpoint(args.output/target,{
                    'gradients':{n:p.grad.detach().cpu() for n,p in model.named_parameters()},
                    'loss':float(total.cpu())/global_count,'real_targets':global_count,
                    'source_cursor':cursor,'raw_gradient_norm':raw_norm})
            norm=finish_gradients(model,scale=1,clip=1.)
        else:
            norm=finish_gradients(model,scale=scale,clip=1.)
        optimizer.step()
        if update==0 and rank==0:
            atomic_checkpoint(args.output/'after-1-model-optimizer.pt',complete_state(model,optimizer))
        work=(model_work(config,depths[update].reshape(-1,2)) if recurrent else
            matmul_ledger(config,1024,32*args.accumulation,recurrent=False)['training_matmul_flops'])
        model_matmuls+=work;optimizer_matmuls+=one_optimizer
        if 20<=update<120:
            warm_tokens+=global_count;warm_model+=work;warm_optimizer+=one_optimizer
        update+=1;cursor+=world*args.accumulation;exposure+=global_count
        if update==1 or update%10==0 or update==121:
            if not reduced_loss:dist.all_reduce(total)
            row={'update':update,'real_targets':exposure,'global_train_nll':float(total.cpu())/global_count,
                'gradient_norm_before_clip':norm,'lr':rate,'elapsed_seconds':time.time()-started}
            rows.append(row)
            if rank==0:print(json.dumps(row),flush=True)
    torch.neuron.synchronize()
    final=complete_state(model,optimizer)
    final_digest=digest(final)
    (args.output/f'replica-{rank}.json').write_text(json.dumps({'full_state_digest':final_digest,
        'fallback_ops':list(torch_neuronx.get_fallback_ops())})+'\n')
    if rank==0:
        atomic_checkpoint(args.output/'after-121-model-optimizer.pt',final)
    dist.barrier()
    if rank==0:
        replicas=[json.loads((args.output/f'replica-{i}.json').read_text()) for i in range(world)]
        if len({r['full_state_digest'] for r in replicas})!=1:
            raise AssertionError('Complete trained model/optimizer replicas differ')
        if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in replicas):
            raise AssertionError('Unexpected complete training fallback')
        result={'status':'CAPTURED_PARCAE_TRAINING_QUALIFICATION','arm':args.arm,
            'accumulation':args.accumulation,'world':world,'precision':args.precision,'mode':args.mode,
            'qualification_only':True,'test_accessed':False,'research_optimizer_updates':0,
            'actual_optimizer_calls_per_replica':update-origin_update,'phase_real_targets':exposure-origin_exposure,
            'total_real_targets':exposure,'cursor':cursor,'model_matmuls':model_matmuls,
            'optimizer_matmuls':optimizer_matmuls,'initial_model_digest':initial['model_digest'],
            'recovered_state_digest':recovered_digest,'complete_state_on_load_exact':args.mode=='replay',
            'per_rank_cpu_rng_on_load_exact':args.mode=='replay',
            'all_replica_model_optimizer_states_equal':True,'final_model_optimizer_digest':final_digest,
            'warm_updates':100 if args.mode=='warm' else 0,'warm_real_targets':warm_tokens,
            'warm_seconds':warm_seconds,'warm_targets_per_second':warm_tokens/warm_seconds if warm_seconds else None,
            'warm_model_matmuls':warm_model,'warm_optimizer_matmuls_all_replicas':warm_optimizer,
            'warm_started_unix':warm_start,'warm_finished_unix':warm_finished,
            'checkpoint_sha256':checkpoint_sha,'checkpoint_state_digest':checkpoint_digest,
            'probe_initial_states_sha256':point_noise_sha,'probe_depths_sha256':point_depth_sha,
            'updated_global_gradient_sha256':file_sha256(args.output/'updated-global-native.pt'),
            'initial_global_gradient_sha256':file_sha256(args.output/'initial-global-native.pt') if args.mode=='warm' else None,
            'first_checkpoint_sha256':file_sha256(args.output/'after-1-model-optimizer.pt') if args.mode=='warm' else None,
            'final_checkpoint_sha256':file_sha256(args.output/'after-121-model-optimizer.pt'),
            'extra_initial_probe_model_matmuls':(model_work(config,depths[0,0]) if recurrent else matmul_ledger(config,1024,32,recurrent=False)['training_matmul_flops']) if args.mode=='warm' else 0,
            'extra_initial_probe_real_targets':sum(sum(packed.counts(routed_indices(0,0,r,32,1,args.accumulation,271828)[0],1)) for r in range(32)) if args.mode=='warm' else 0,
            'checkpoint_seconds':checkpoint_seconds,'progress':rows,'signature':signature,
            'wall_seconds':time.time()-started,
            'scope':'Complete qualification updates, natural packing/H2D, per-sequence depth, no_sync normalization, manual unscale, finite clipping and two-family optimizer. Warm100 updates exclude initial probe/compilation, final checkpoint and updated-point/replay numerical audits. Extra overheads/failed attempts remain in phase receipts. No scientific learning claim or budget authorization.'}
        (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
