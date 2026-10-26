"""Zero-update full-global capture with a separate precision-probe envelope."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import time

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from compute_trainium.depth_mixture import routed_indices
from compute_trainium.future_span import FutureSpanConfig
from compute_trainium import future_span_gradient_point as points
from compute_trainium.future_span_training import FutureSpanObjective,objective_coefficients,training_window
from compute_trainium.mixing_control_science import signature
from compute_trainium.mixing_precision_probe import PrecisionMixingProbe
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_mixing_control import mixing_only_ledger
from compute_trainium.parcae_training_candidate import finish_gradients,map_after_shared_validation
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--variant',choices=['mix32','mixc32','full32'],required=True)
    p.add_argument('--comm-port',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    a=p.parse_args();started=time.time()
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Diagnostic plan changed')
    probe=json.loads(a.plan.read_text())
    if probe['purpose']!='zero_update_mixing_precision_diagnosis' or probe['root_comm_ports'].get(a.output.name)!=a.comm_port:
        raise ValueError('Exact registered diagnostic phase required')
    repo=Path(__file__).resolve().parents[1]
    if any(file_sha256(repo/f)!=sha for f,sha in probe['worker_source_sha256'].items()):raise AssertionError('Frozen diagnostic worker changed')
    point=probe['point'];parent_file=a.root/point['parent_plan'];checkpoint=a.root/point['checkpoint']
    if file_sha256(parent_file)!=point['parent_plan_sha256']:raise AssertionError('Original trained plan changed')
    parent=json.loads(parent_file.read_text());case=parent['cases'][point['case']]
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron');rank,world=dist.get_rank(),dist.get_world_size()
    if world!=32:raise ValueError('Complete32-rank objective required')
    if rank==0 and file_sha256(checkpoint)!=point['checkpoint_sha256']:raise AssertionError('Failed model checkpoint changed')
    dist.barrier();saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    counts,anchors,depths,work=points.point_tables(a.root,parent_file,parent,case);update,cursor=saved['update'],saved['cursor']
    points.signature=signature
    points.validate_point(saved,point['parent_plan_sha256'],parent,point['case'],counts[cursor]-counts[0],anchors[cursor],work[:update].sum())
    complete=digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})
    if complete!=saved['model_optimizer_digest']:raise AssertionError('Complete original model/optimizer readback differs')
    c=ParcaeCandidateConfig(**parent['model']);span=FutureSpanConfig(**parent['future_config'])
    model=PrecisionMixingProbe(c,span,precision_mode=a.variant,initialize=False,fp16_body=True);model.load_state_dict(saved['model'])
    before=digest(dict(sorted(model.state_dict().items())));del saved
    if rank==0:
        checked=PackedCorpus.from_mapped(a.root/parent['data']['packed'],1024,23,parent['data']['corpus_sha256'])
        if checked.fingerprint!=parent['data']['packing_fingerprint']:raise AssertionError('Natural packing changed')
        del checked
    dist.barrier();packed=map_after_shared_validation(a.root/parent['data']['packed'],parent['data']['manifest_sha256'])
    total=sum(packed.counts(cursor,512));anchor_count=int(anchors[cursor+512]-anchors[cursor])
    coefficients=objective_coefficients(total,anchor_count,case['objective_weight'],32).to('neuron')
    model.to('neuron');torch._dynamo.config.optimize_ddp=False;model.compile_stages('neuron')
    ddp=DDP(FutureSpanObjective(model,backend='neuron'),device_ids=None,broadcast_buffers=False,bucket_cap_mb=64,gradient_as_bucket_view=True)
    a.output.mkdir(exist_ok=True);totals=torch.zeros(2,device='neuron');rows=[];noise=[]
    for micro in range(16):
        window=routed_indices(cursor,micro,rank,32,1,16,parent['routing_seed'])[0]
        values=training_window(packed,window,c,span,case['state_seed']);n,k=map(int,depths[update,micro,rank])
        if (n,k)!=(2,2):raise ValueError('Original fixed-two-mix objective required')
        rows.append(dict(window=window,micro=micro,rank=rank,n=n,k=k,h0_digest=digest(values[3]),valid_anchors=int(values[-1].sum())))
        noise.append(values[3]);transferred=[v.contiguous().to('neuron') for v in values]
        with ddp.no_sync() if micro<15 else contextlib.nullcontext():
            value,parts=ddp(*transferred,coefficients,n,k);(value*8192).backward()
        totals+=parts
    finish_gradients(model,scale=8192,clip=float('inf'));dist.all_reduce(totals)
    gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters()}
    if any(not torch.isfinite(g).all() for g in gradients.values()):raise FloatingPointError('Nonfinite diagnostic gradient')
    after=digest(dict(sorted(model.state_dict().items())))
    if before!=after:raise AssertionError('Diagnostic changed trained model')
    noise_file=a.output/f'h0-rank{rank}.npy';np.save(noise_file,torch.cat(noise).numpy())
    replica=dict(model_digest=after,gradient_digest=digest(gradients),rows=rows,
                 noise_sha256=file_sha256(noise_file),fallback_ops=list(torch_neuronx.get_fallback_ops()))
    (a.output/f'rank{rank}.json').write_text(json.dumps(replica)+'\n');dist.barrier()
    if rank==0:
        replicas=[json.loads((a.output/f'rank{i}.json').read_text()) for i in range(32)]
        if len({v['model_digest'] for v in replicas})!=1 or len({v['gradient_digest'] for v in replicas})!=1:raise AssertionError('Diagnostic replicas differ')
        if any(set(v['fallback_ops'])-{'aten::_to_copy'} for v in replicas):raise AssertionError('Unexpected diagnostic fallback')
        rows=sorted([v for replica in replicas for v in replica['rows']],key=lambda v:v['window'])
        if [v['window'] for v in rows]!=list(range(cursor,cursor+512)) or sum(v['valid_anchors'] for v in rows)!=anchor_count:
            raise AssertionError('Original complete objective coverage differs')
        vector=a.output/'gradient.pt';atomic_checkpoint(vector,dict(gradients=gradients,ntp_nll=float(totals[0].cpu())/total,future_mean=float(totals[1].cpu())/max(1,anchor_count)))
        matrix=sum(mixing_only_ledger(c,1024,total_steps=v['n'],backward_cap=v['k'])['training_matmul_flops'] for v in rows)+512*6*len(span.positions)*c.dim*c.vocab_size
        report=dict(status='CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT',case_id=point['case'],cell=case['cell'],
            plan_sha256=point['parent_plan_sha256'],probe_plan_sha256=a.plan_sha,precision_override=a.variant,
            checkpoint_sha256=point['checkpoint_sha256'],full_state_digest=complete,training_update=update,source_cursor=cursor,
            real_targets=total,valid_anchors=anchor_count,windows=512,rows=rows,gradient_sha256=file_sha256(vector),
            model_unchanged=True,noise_sha256_by_rank=[v['noise_sha256'] for v in replicas],replica_gradients_equal=True,
            diagnostic_matrix_ops=matrix,optimizer_updates=0,research_optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,
            scope='Explicit precision override at an unchanged trained checkpoint. Parent plan identifies its training history;probe plan identifies this zero-update execution. No scientific fit or qualification of future optimizer updates.')
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
