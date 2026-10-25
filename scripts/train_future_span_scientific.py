"""Finite frozen scientific segments for the prospective four-cell objective test."""
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
from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig
from compute_trainium.future_span_scientific_state import (
    SCIENCE,INTEGRATION,matrix_work,signature,worker_sources,validate_initial,validate_checkpoint,
)
from compute_trainium.future_span_training import FutureSpanObjective,objective_coefficients,training_window
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer
from compute_trainium.parcae_scientific_state import configure_optimizer,schedule_identity
from compute_trainium.parcae_training_candidate import finish_gradients,map_after_shared_validation
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def complete_state(model,optimizer):
    return {'model':dict(sorted(model.state_dict().items())),'optimizer':optimizer.state_dict()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--case',required=True);parser.add_argument('--segment-end',type=int,required=True)
    parser.add_argument('--comm-port',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path);parser.add_argument('--resume-sha');parser.add_argument('--corpus')
    args=parser.parse_args();started=time.time()
    if file_sha256(args.plan)!=args.plan_sha:raise AssertionError('Frozen scientific future plan changed')
    plan=json.loads(args.plan.read_text());case=plan['cases'][args.case]
    if plan['purpose'] not in [SCIENCE,INTEGRATION] or not plan['launch_gates_complete']:
        raise ValueError('A qualified registered future-span plan is required')
    qualification=plan['purpose']==INTEGRATION
    if qualification and case['total_updates']>40:raise ValueError('Driver qualification is capped at40 cumulative updates')
    if not 0<args.segment_end<=case['total_updates'] or args.segment_end not in case['segment_ends']:
        raise ValueError('Unregistered finite source segment')
    if (args.resume is None)!=(args.resume_sha is None):raise ValueError('Checksum-qualified resume required')
    if (plan['precision']!='fp16_body_fp32_heads_masters' or plan['accumulation']!=16
            or plan['future_config']['context']!=1024):raise ValueError('Only the qualified geometry/precision is eligible')
    if plan['root_comm_ports'].get(args.output.name)!=args.comm_port:raise ValueError('Exact owned phase/port required')
    code_root=Path(__file__).resolve().parents[1]
    if any(file_sha256(code_root/p)!=sha for p,sha in worker_sources(plan).items()):raise AssertionError('Frozen worker source changed')
    current_signature=signature(args.plan_sha,plan,case)
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(args.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=32:raise ValueError('Exactly32 ranks on one eight-chip half required')
    args.output.mkdir(exist_ok=True)
    experiment=args.plan.parent;data=plan['data'];packing=args.root/data['packed']
    aux=args.root/plan['auxiliary_data']['directory'];aux_data=plan['auxiliary_data']
    initial=plan['initial_states'][case['initial_state']]
    initial_file=experiment/initial['file'];depth_file=experiment/case['depth_table'];work_file=experiment/case['work_table']
    if rank==0:
        for path,sha in [(initial_file,initial['sha256']),(depth_file,case['depth_table_sha256']),
                         (work_file,case['work_table_sha256']),(aux/aux_data['prefix_file'],aux_data['prefix_sha256'])]:
            if file_sha256(path)!=sha:raise AssertionError('Scientific initializer/depth/work/auxiliary counts changed')
        verified=PackedCorpus.from_mapped(packing,1024,23,data['corpus_sha256'])
        if verified.fingerprint!=data['packing_fingerprint']:raise AssertionError('Original natural source packing changed')
        del verified
    dist.barrier();packed=map_after_shared_validation(packing,data['manifest_sha256'])
    depth=np.load(depth_file,mmap_mode='r');work=np.load(work_file,mmap_mode='r')
    anchor_prefix=np.load(aux/aux_data['prefix_file'],mmap_mode='r')
    if (depth.shape[1:]!=(16,32,2) or len(work)!=len(depth) or len(work)<=case['total_updates']
            or len(anchor_prefix)<=case['total_updates']*512 or not np.array_equal(depth[...,0],depth[...,1])
            or (case['arm']=='one' and not np.all(depth==1))):raise AssertionError('Provisioned full-gradient/source geometry differs')
    saved=torch.load(initial_file,map_location='cpu',mmap=True,weights_only=False);validate_initial(saved,plan,case)
    config=ParcaeCandidateConfig(**plan['model']);span=FutureSpanConfig(**plan['future_config'])
    torch.manual_seed(case['seed'])
    model=FutureSpanCandidate(config,span,head_seed=initial['head_seed'],initialize=False,fp16_body=True)
    model.load_state_dict(saved['model'],strict=True);del saved
    if digest(dict(sorted(model.state_dict().items())))!=initial['model_digest']:raise AssertionError('Fresh shared initialization differs')
    if sum(p.numel() for p in model.parameters())!=case['training_parameters']:raise AssertionError('Training parameter inventory differs')
    model.to('neuron');optimizer=ParcaeRuntimeScalarOptimizer(model,gate_policy='release_muon',
        lr=case['peak_lr'],muon_lr=case['peak_lr'],fused_adamw=True,muon_backend=None)
    optimizer_ops=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')
    expected_work=matrix_work(config,span,depth,world=32,optimizer_per_replica=optimizer_ops)
    if not np.array_equal(work,expected_work):raise AssertionError('Scientific dense matrix-work table is incorrect')
    if sum(map(int,work))>np.iinfo(np.int64).max:raise OverflowError('Cumulative matrix-work ledger exceeds signed64')
    prefix_work=work.cumsum(dtype=np.int64)
    update=cursor=exposure=anchors=matrix_ops=0;domains=[0,0,0];recovered_digest=None;recovered_rng=None
    if args.resume:
        if rank==0 and file_sha256(args.resume)!=args.resume_sha:raise AssertionError('Scientific resume checksum changed')
        dist.barrier();saved=torch.load(args.resume,map_location='cpu',mmap=True,weights_only=False)
        expected_domains=packed.counts(0,saved['cursor'])
        validate_checkpoint(saved,args.case,case,current_signature,expected_targets=sum(expected_domains),
            expected_domains=expected_domains,expected_matrix_ops=int(prefix_work[saved['update']-1]),
            expected_anchors=int(anchor_prefix[saved['cursor']]))
        if saved['update']>=args.segment_end:raise ValueError('This finite segment is already complete')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        recovered_digest=digest(complete_state(model,optimizer))
        if recovered_digest!=saved['model_optimizer_digest']:raise AssertionError('Exact model/optimizer recovery failed')
        update,cursor,exposure,anchors,matrix_ops=(saved[k] for k in ['update','cursor','real_targets','valid_anchors','estimated_dense_matrix_ops'])
        domains=list(saved['domain_targets']);recovered_rng=saved['cpu_rng_by_rank'][rank].clone();del saved
    origin_update,origin_exposure,origin_anchors,origin_matrix=update,exposure,anchors,matrix_ops
    torch._dynamo.config.optimize_ddp=False;model.compile_stages('neuron')
    ddp=DDP(FutureSpanObjective(model,backend='neuron'),device_ids=None,broadcast_buffers=False,
            bucket_cap_mb=64,gradient_as_bucket_view=True)
    if recovered_rng is not None:
        torch.set_rng_state(recovered_rng)
        if not torch.equal(torch.get_rng_state(),recovered_rng):raise AssertionError('Exact per-rank CPU RNG recovery failed')
    progress=[];checkpoints=[];first_probe=None

    def checkpoint():
        torch.save(torch.get_rng_state(),args.output/f'rng-u{update}-rank{rank}.pt');dist.barrier()
        state_digest=digest(complete_state(model,optimizer))
        (args.output/f'replica-u{update}-rank{rank}.json').write_text(json.dumps({'full_state_digest':state_digest,
            'fallback_ops':list(torch_neuronx.get_fallback_ops())})+'\n');dist.barrier()
        if rank==0:
            replicas=[json.loads((args.output/f'replica-u{update}-rank{i}.json').read_text()) for i in range(32)]
            if len({r['full_state_digest'] for r in replicas})!=1:raise AssertionError('Full scientific replicas differ')
            if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in replicas):raise AssertionError('Unexpected scientific fallback')
            path=args.output/f'checkpoint-u{update}.pt'
            atomic_checkpoint(path,{**complete_state(model,optimizer),'purpose':plan['purpose'],
                'qualification_only':qualification,'case_id':args.case,'signature':current_signature,
                'schedule':schedule_identity(case),'stable_trunk':exposure<=case['cooldown_start_targets'],
                'update':update,'cursor':cursor,'real_targets':exposure,'domain_targets':domains,
                'valid_anchors':anchors,'future_source_position_presentations':anchors*(span.horizon-1),
                'computed_binary_presence_terms':anchors*config.vocab_size,'executed_head_slots':cursor*len(span.positions),
                'estimated_dense_matrix_ops':matrix_ops,'model_optimizer_digest':state_digest,
                'cpu_rng_by_rank':[torch.load(args.output/f'rng-u{update}-rank{i}.pt',weights_only=True) for i in range(32)],
                'native_rng_ops_used':False,'rng_policy':'Explicit CPU source-window latent state and immutable exogenous depth table.'})
            row={'update':update,'checkpoint_sha256':file_sha256(path),'full_state_digest':state_digest,
                'real_targets':exposure,'valid_anchors':anchors,'estimated_dense_matrix_ops':matrix_ops,
                'all_complete_replica_states_equal':True}
            checkpoints.append(row);print(json.dumps({'event':'checkpoint',**row}),flush=True)
        dist.barrier();gc.collect()

    while update<args.segment_end:
        counts=packed.counts(cursor,512);total=sum(counts)
        auxiliary=int(anchor_prefix[cursor+512]-anchor_prefix[cursor])
        if not 0<=auxiliary<=512*len(span.positions):raise AssertionError('Invalid global future-anchor count')
        coefficients=objective_coefficients(total,auxiliary,case['objective_weight'],32).to('neuron')
        schedule=configure_optimizer(optimizer,case,exposure+total);optimizer.zero_grad(set_to_none=True)
        probe_first=qualification and update==0
        log=probe_first or (update+1)%10==0 or update+1==args.segment_end
        totals=torch.zeros(2,device='neuron') if log else None
        first_rows=[]
        for micro in range(16):
            window=routed_indices(cursor,micro,rank,32,1,16,plan['routing_seed'])[0]
            values=training_window(packed,window,config,span,case['state_seed'])
            n,k=map(int,depth[update,micro,rank])
            if probe_first:first_rows.append({'window':window,'micro':micro,'rank':rank,'n':n,'k':k,
                'h0_digest':digest(values[3]),'valid_anchors':int(values[-1].sum())})
            transferred=[v.contiguous().to('neuron') for v in values]
            with ddp.no_sync() if micro<15 else contextlib.nullcontext():
                value,parts=ddp(*transferred,coefficients,n,k);(value*8192).backward()
            if log:totals+=parts
        norm=finish_gradients(model,scale=8192,clip=float('inf') if probe_first else 1.)
        if probe_first:
            dist.all_reduce(totals)
            vectors={name:p.grad.detach().cpu() for name,p in model.named_parameters()}
            (args.output/f'first-gradient-rank{rank}.json').write_text(json.dumps({
                'gradient_digest':digest(vectors),'rows':first_rows})+'\n');dist.barrier()
            if rank==0:
                replicas=[json.loads((args.output/f'first-gradient-rank{i}.json').read_text()) for i in range(32)]
                if len({r['gradient_digest'] for r in replicas})!=1:raise AssertionError('Driver first global gradients differ across replicas')
                rows=sorted([row for r in replicas for row in r['rows']],key=lambda r:r['window'])
                if [r['window'] for r in rows]!=list(range(512)) or sum(r['valid_anchors'] for r in rows)!=auxiliary:
                    raise AssertionError('Driver first global source/auxiliary coverage differs')
                file=args.output/'first-global-gradient.pt'
                atomic_checkpoint(file,{'gradients':vectors,'ntp_nll':float(totals[0].cpu())/total,
                    'future_mean':float(totals[1].cpu())/max(1,auxiliary)})
                first_probe={'rows':rows,'source_cursor':0,'real_targets':total,'valid_anchors':auxiliary,
                    'gradient_file':file.name,'gradient_sha256':file_sha256(file),'raw_gradient_norm':norm}
            del vectors;finish_gradients(model,scale=1,clip=1.)
        optimizer.step()
        if probe_first and rank==0:atomic_checkpoint(args.output/'after-first-update.pt',complete_state(model,optimizer))
        update+=1;cursor+=512;exposure+=total;anchors+=auxiliary;matrix_ops+=int(work[update-1])
        domains=[a+b for a,b in zip(domains,counts,strict=True)]
        if log:
            if not probe_first:dist.all_reduce(totals)
            ntp=float(totals[0].cpu())/total;future=float(totals[1].cpu())/max(1,auxiliary)
            if not np.isfinite([ntp,future]).all():raise FloatingPointError('Nonfinite scientific objective;no skipped update')
            row={'update':update,'real_targets':exposure,'valid_anchors':anchors,'estimated_dense_matrix_ops':matrix_ops,
                'global_train_nll':ntp,'global_future_loss':future,'combined_objective':ntp+case['objective_weight']*future,
                'gradient_norm_before_clip':norm,**schedule,'elapsed_seconds':time.time()-started}
            if rank==0:progress.append(row);print(json.dumps(row),flush=True)
        if update in case['checkpoint_updates'] or update==args.segment_end:checkpoint()
    if (cursor!=update*512 or domains!=packed.counts(0,cursor) or exposure!=sum(domains)
            or anchors!=int(anchor_prefix[cursor]) or matrix_ops!=int(prefix_work[update-1])):
        raise AssertionError('Final source/auxiliary/matrix counters differ')
    if rank==0:
        report={'status':'COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT','purpose':plan['purpose'],
            'case_id':args.case,'plan_sha256':args.plan_sha,'cell':case['cell'],'arm':case['arm'],
            'objective_weight':case['objective_weight'],'seed':case['seed'],'update':update,'cursor':cursor,
            'real_targets':exposure,'domain_targets':domains,'valid_anchors':anchors,'estimated_dense_matrix_ops':matrix_ops,
            'segment_real_targets':exposure-origin_exposure,'segment_valid_anchors':anchors-origin_anchors,
            'segment_estimated_dense_matrix_ops':matrix_ops-origin_matrix,'training_parameters':case['training_parameters'],
            'deployed_parameters':config.parameter_count(True),'actual_optimizer_calls_per_replica':update-origin_update,
            'complete_state_on_load_exact':args.resume is not None,'recovered_state_digest':recovered_digest,
            'cpu_rng_on_load_exact':args.resume is not None,'checkpoint_records':checkpoints,'progress':progress,
            'first_global_probe':first_probe,
            'first_update_sha256':file_sha256(args.output/'after-first-update.pt') if first_probe is not None else None,
            'qualification_only':qualification,'research_optimizer_updates':0 if qualification else update-origin_update,
            'test_accessed':False,'wall_seconds':time.time()-started,
            'scope':'Finite registered future-span scientific segment. Same original real targets and source order;separate global auxiliary normalization. Dense matrix counts include every head slot and replica Muon work,not hardware padding or chip-time equivalence. No quality or significance claim without independent readouts.'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
