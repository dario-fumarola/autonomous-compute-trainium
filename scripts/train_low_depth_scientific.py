"""Finite frozen-plan scientific segments using the qualified Parcae operators.

No default run, data download, automatic budget extension or qualification-weight
reuse. This entry point remains unlaunched until its protocol/qualification gates.
"""
import argparse
import contextlib
import gc
import json
import os
import time
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from compute_trainium.depth_mixture import routed_indices
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_fp16_candidate import ParcaeFP16Candidate
from compute_trainium.parcae_optimizer_precise import ParcaePreciseOptimizer
from compute_trainium.parcae_scientific_schedule import update_work
from compute_trainium.parcae_scientific_state import (
    signature,schedule_identity,validate_checkpoint,configure_optimizer,worker_sources,
)
from compute_trainium.parcae_stages_candidate import ParcaeStagesCandidate
from compute_trainium.parcae_training_candidate import initial_state,finish_gradients,map_after_shared_validation
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def complete_state(model,optimizer):
    return {'model':dict(sorted(model.state_dict().items())),'optimizer':optimizer.state_dict()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus') # Common isolated launcher; frozen plan governs the actual path.
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--segment-end',type=int,required=True)
    parser.add_argument('--comm-port',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resume-sha')
    args=parser.parse_args()
    if not 1024<=args.comm_port<65536:raise ValueError('Explicit unprivileged root communication port required')
    if file_sha256(args.plan)!=args.plan_sha:raise AssertionError('Bootstrap plan changed')
    bootstrap=json.loads(args.plan.read_text())
    if bootstrap['root_comm_ports'].get(args.output.name)!=args.comm_port:
        raise ValueError('Communication port is not registered for this exact phase')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(args.comm_port)
    import torch_neuronx
    from scripts.train_parcae_precise_qualification import Objective
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=32 or (args.resume is None)!=(args.resume_sha is None):
        raise ValueError('Frozen32-rank scientific segment and checksum-qualified resume required')
    if file_sha256(args.plan)!=args.plan_sha:raise AssertionError('Scientific plan changed')
    plan=json.loads(args.plan.read_text())
    qualification=plan['purpose']=='low_depth_driver_integration_qualification'
    if plan['purpose'] not in ['natural_low_depth_scientific_pilot','low_depth_driver_integration_qualification'] or not plan['launch_gates_complete']:
        raise ValueError('A registered qualified scientific plan is required')
    code_root=Path(__file__).resolve().parents[1]
    if any(file_sha256(code_root/p)!=sha for p,sha in worker_sources(plan).items()):
        raise AssertionError('Frozen scientific implementation changed')
    case=plan['cases'][args.case]
    if qualification and case['total_updates']>40:raise ValueError('Driver qualification is bounded to40 cumulative updates')
    if not 0<args.segment_end<=case['total_updates'] or args.segment_end not in case['segment_ends']:
        raise ValueError('Unregistered scientific segment boundary')
    if plan['precision'] not in ['fp32','fp16'] or plan['accumulation'] not in [8,16]:
        raise ValueError('Only the qualified precision/batch geometries are eligible')
    args.output.mkdir(exist_ok=True)
    started=time.time();experiment=args.plan.parent
    data=plan['data'];corpus=args.root/data['corpus'];directory=args.root/data['packed']
    if rank==0:
        if file_sha256(corpus)!=data['corpus_sha256']:raise AssertionError('Natural corpus changed')
        verified=PackedCorpus.from_mapped(directory,1024,23,data['corpus_sha256'])
        if verified.fingerprint!=data['packing_fingerprint']:raise AssertionError('Natural packing changed')
        del verified
    dist.barrier()
    packed=map_after_shared_validation(directory,data['manifest_sha256'])
    initial=plan['initial_states'][case['initial_state']]
    if rank==0 and file_sha256(experiment/initial['file'])!=initial['sha256']:
        raise AssertionError('Fresh scientific initialization changed')
    dist.barrier()
    saved=torch.load(experiment/initial['file'],map_location='cpu',mmap=True,weights_only=False)
    if saved['qualification_only'] is not qualification or saved.get('seed',saved.get('model_seed'))!=case['seed']:
        raise ValueError('Scientific fits require their own fresh random initialization')
    config=ParcaeCandidateConfig(**plan['model']);recurrent=case['arm']!='ordinary'
    cls=ParcaeFP16Candidate if plan['precision']=='fp16' else ParcaeStagesCandidate
    torch.manual_seed(case['seed']);model=cls(config,recurrent=recurrent,initialize=False)
    model.load_state_dict(saved['model'])
    if digest(dict(sorted(model.state_dict().items())))!=initial['model_digest']:
        raise AssertionError('Scientific initial state does not match its manifest')
    del saved
    if sum(p.numel() for p in model.parameters())!=case['parameters']:raise AssertionError('Model inventory changed')
    model.to('neuron')
    optimizer=ParcaePreciseOptimizer(model,gate_policy='release_muon',lr=case['peak_lr'],
                                   muon_lr=case['peak_lr'],fused_adamw=True,muon_backend=None)
    table_file=experiment/case['depth_table'];work_file=experiment/case['work_table']
    if rank==0 and (file_sha256(table_file)!=case['depth_table_sha256'] or file_sha256(work_file)!=case['work_table_sha256']):
        raise AssertionError('Executed depth/work plan changed')
    dist.barrier();depths=np.load(table_file,mmap_mode='r');work=np.load(work_file,mmap_mode='r')
    accumulation=plan['accumulation'];windows=32*accumulation
    if depths.shape[1:]!=(accumulation,32,2) or len(depths)<=case['total_updates'] or len(work)!=len(depths):
        raise AssertionError('Provisioned execution/work table shape differs')
    one_optimizer=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')
    # Check the complete work table against actual model/head/optimizer shapes.
    expected_work=update_work(config,depths,recurrent=recurrent,context=1024,world=32,
                              optimizer_per_replica=one_optimizer)
    if not np.array_equal(expected_work,work):raise AssertionError('Scientific executed-work table is incorrect')
    prefix_work=np.cumsum(work,dtype=np.int64)
    current_signature=signature(args.plan_sha,plan,case)
    update=cursor=exposure=matmuls=0;domain_counts=[0,0,0];recovered_digest=None;recovered_rng=None
    if args.resume:
        if rank==0 and file_sha256(args.resume)!=args.resume_sha:raise AssertionError('Resume checkpoint file changed')
        dist.barrier();saved=torch.load(args.resume,map_location='cpu',mmap=True,weights_only=False)
        expected_targets=sum(packed.counts(0,saved['cursor']))
        expected_matmuls=int(prefix_work[saved['update']-1]) if saved['update'] else 0
        validate_checkpoint(saved,args.case,case,current_signature,expected_targets=expected_targets,expected_matmuls=expected_matmuls)
        if saved['update']>=args.segment_end:raise ValueError('Segment is already complete; do not retrain it')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        recovered_digest=digest(complete_state(model,optimizer))
        if recovered_digest!=saved['model_optimizer_digest']:raise AssertionError('Scientific full state changed on load')
        update,cursor,exposure,matmuls=saved['update'],saved['cursor'],saved['real_targets'],saved['executed_matmuls']
        domain_counts=list(saved['domain_targets']);recovered_rng=saved['cpu_rng_by_rank'][rank].clone()
        if domain_counts!=list(packed.counts(0,cursor)):raise AssertionError('Recovered domain/source counts differ')
        del saved
    origin_update,origin_exposure,origin_matmuls=update,exposure,matmuls
    torch._dynamo.config.optimize_ddp=False;model.compile_stages('neuron')
    objective=Objective(model);ddp=DDP(objective,device_ids=None,broadcast_buffers=False,bucket_cap_mb=64,gradient_as_bucket_view=True)
    if recovered_rng is not None:
        torch.set_rng_state(recovered_rng)
        if not torch.equal(torch.get_rng_state(),recovered_rng):raise AssertionError('Scientific per-rank CPU RNG restore differs')
    scale=8192 if plan['precision']=='fp16' else 1
    progress=[];checkpoint_records=[]

    def batch(global_update,micro,worker=rank):
        window=routed_indices(global_update*windows,micro,worker,32,1,accumulation,plan['routing_seed'])[0]
        item=packed.batch([window],0)[0]
        state=initial_state(config,window,case['state_seed']) if recurrent else None
        n,k=map(int,depths[global_update,micro,worker])
        values=[t.contiguous().to('neuron') for t in [item.inputs,item.targets,item.target_mask]]
        return (*values,state.contiguous().to('neuron') if recurrent else None,n,k)

    def checkpoint():
        torch.save(torch.get_rng_state(),args.output/f'cpu-rng-u{update}-rank{rank}.pt');dist.barrier()
        state=complete_state(model,optimizer);state_digest=digest(state)
        (args.output/f'replica-u{update}-rank{rank}.json').write_text(json.dumps({'full_state_digest':state_digest,
            'fallback_ops':list(torch_neuronx.get_fallback_ops())})+'\n')
        dist.barrier()
        if rank==0:
            replicas=[json.loads((args.output/f'replica-u{update}-rank{i}.json').read_text()) for i in range(32)]
            if len({r['full_state_digest'] for r in replicas})!=1:raise AssertionError('Scientific complete replica states differ')
            if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in replicas):raise AssertionError('Unexpected scientific fallback')
            file=args.output/f'checkpoint-u{update}.pt'
            atomic_checkpoint(file,{**state,'model_optimizer_digest':state_digest,
                'purpose':plan['purpose'],'qualification_only':qualification,'case_id':args.case,'signature':current_signature,
                'schedule':schedule_identity(case),'stable_trunk':update<=case['cooldown_start_update'],
                'update':update,'cursor':cursor,'real_targets':exposure,'domain_targets':domain_counts,
                'executed_matmuls':matmuls,'cpu_rng_by_rank':[torch.load(args.output/f'cpu-rng-u{update}-rank{i}.pt',weights_only=True) for i in range(32)],
                'framework_native_rng_ops_used':False,'rng_policy':'Explicit CPU state key per absolute source window; frozen sampled-depth table.'})
            row={'update':update,'checkpoint_sha256':file_sha256(file),'full_state_digest':state_digest,
                'real_targets':exposure,'executed_matmuls':matmuls,'all_complete_replica_states_equal':True}
            checkpoint_records.append(row);print(json.dumps({'event':'checkpoint',**row}),flush=True)
        dist.barrier();del state;gc.collect()

    for _ in range(args.segment_end-update):
        counts=list(packed.counts(cursor,windows));global_targets=sum(counts)
        rate=configure_optimizer(optimizer,case,exposure+global_targets)
        optimizer.zero_grad(set_to_none=True);log=(update+1)%10==0 or update+1==args.segment_end
        total=torch.zeros((),device='neuron') if log else None
        for micro in range(accumulation):
            values=batch(update,micro)
            with ddp.no_sync() if micro<accumulation-1 else contextlib.nullcontext():
                value=ddp(*values);(value*(world/global_targets)*scale).backward()
            if log:total+=value.detach()
        norm=finish_gradients(model,scale=scale,clip=1.);optimizer.step()
        update+=1;cursor+=windows;exposure+=global_targets;matmuls+=int(work[update-1])
        domain_counts=[a+b for a,b in zip(domain_counts,counts,strict=True)]
        if log:
            dist.all_reduce(total)
            row={'update':update,'real_targets':exposure,'executed_matmuls':matmuls,
                 'global_train_nll':float(total.cpu())/global_targets,'gradient_norm_before_clip':norm,
                 **rate,'elapsed_seconds':time.time()-started}
            if not np.isfinite(row['global_train_nll']):raise FloatingPointError('Nonfinite scientific training loss')
            progress.append(row)
            if rank==0:print(json.dumps(row),flush=True)
        if update in case['checkpoint_updates'] or update==args.segment_end:checkpoint()
    if cursor!=update*windows or exposure!=sum(packed.counts(0,cursor)) or matmuls!=int(prefix_work[update-1]):
        raise AssertionError('Scientific final source/work counters disagree')
    if rank==0:
        result={'status':'COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT','case_id':args.case,
            'purpose':plan['purpose'],'plan_sha256':args.plan_sha,'arm':case['arm'],'seed':case['seed'],
            'precision':plan['precision'],'accumulation':accumulation,'world':32,'parameters':case['parameters'],
            'initial_model_digest':initial['model_digest'],'actual_optimizer_calls_per_replica':update-origin_update,
            'segment_real_targets':exposure-origin_exposure,'segment_executed_matmuls':matmuls-origin_matmuls,
            'update':update,'source_cursor':cursor,'real_targets':exposure,'domain_targets':domain_counts,
            'executed_matmuls':matmuls,'complete_state_on_load_exact':args.resume is not None,
            'recovered_state_digest':recovered_digest,'checkpoint_records':checkpoint_records,
            'progress':progress,'test_accessed':False,'wall_seconds':time.time()-started,
            'qualification_only':qualification,'research_optimizer_updates':0 if qualification else update-origin_update,
            'scope':'Finite frozen scientific source segment with real-target objective, declared schedule and exact full state checkpoints. Held-out quality, independent audits, total campaign costs and scientific conclusions are separate.'}
        (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
