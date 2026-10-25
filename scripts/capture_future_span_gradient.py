"""Zero-update full512-window objective at a registered scientific state."""
import argparse
import contextlib
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
from compute_trainium.future_span_gradient_point import point_tables,validate_point
from compute_trainium.future_span_training import FutureSpanObjective,objective_coefficients,training_window
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_training_candidate import finish_gradients,map_after_shared_validation
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--case',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha',required=True);p.add_argument('--comm-port',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    a=p.parse_args();started=time.time()
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen scientific probe plan changed')
    plan=json.loads(a.plan.read_text());case=plan['cases'][a.case]
    if plan['root_comm_ports'].get(a.output.name)!=a.comm_port:raise ValueError('Owned registered probe port required')
    code_root=Path(__file__).resolve().parents[1]
    if any(file_sha256(code_root/f)!=sha for f,sha in plan['worker_source_sha256'].items()):
        raise AssertionError('Exact saved probe source changed')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=32:raise ValueError('Exactly32 ranks required')
    if rank==0 and file_sha256(a.checkpoint)!=a.checkpoint_sha:raise AssertionError('Registered checkpoint changed')
    dist.barrier();saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    counts,anchors,depths,work=point_tables(a.root,a.plan,plan,case);update,cursor=saved['update'],saved['cursor']
    validate_point(saved,a.plan_sha,plan,a.case,counts[cursor]-counts[0],anchors[cursor],work[:update].sum())
    complete=digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})
    if complete!=saved['model_optimizer_digest']:raise AssertionError('Full scientific state readback differs')
    config=ParcaeCandidateConfig(**plan['model']);span=FutureSpanConfig(**plan['future_config'])
    model=FutureSpanCandidate(config,span,initialize=False,fp16_body=True);model.load_state_dict(saved['model'])
    before=digest(dict(sorted(model.state_dict().items())));del saved
    if update>=len(depths):raise ValueError('Next full TRAIN depth row was not provisioned')
    if rank==0:
        verified=PackedCorpus.from_mapped(a.root/plan['data']['packed'],1024,23,plan['data']['corpus_sha256'])
        if verified.fingerprint!=plan['data']['packing_fingerprint']:raise AssertionError('Natural packing differs')
        del verified
    dist.barrier();packed=map_after_shared_validation(a.root/plan['data']['packed'],plan['data']['manifest_sha256'])
    total=sum(packed.counts(cursor,512));anchor_count=int(anchors[cursor+512]-anchors[cursor])
    coefficients=objective_coefficients(total,anchor_count,case['objective_weight'],32).to('neuron')
    model.to('neuron');torch._dynamo.config.optimize_ddp=False;model.compile_stages('neuron')
    ddp=DDP(FutureSpanObjective(model,backend='neuron'),device_ids=None,broadcast_buffers=False,
            bucket_cap_mb=64,gradient_as_bucket_view=True)
    a.output.mkdir(exist_ok=True);totals=torch.zeros(2,device='neuron');rows=[];noise=[]
    for micro in range(16):
        window=routed_indices(cursor,micro,rank,32,1,16,plan['routing_seed'])[0]
        values=training_window(packed,window,config,span,case['state_seed']);n,k=map(int,depths[update,micro,rank])
        if n!=k:raise ValueError('Every sampled recurrent step must receive gradients')
        rows.append({'window':window,'micro':micro,'rank':rank,'n':n,'k':k,
                     'h0_digest':digest(values[3]),'valid_anchors':int(values[-1].sum())});noise.append(values[3])
        transferred=[v.contiguous().to('neuron') for v in values]
        with ddp.no_sync() if micro<15 else contextlib.nullcontext():
            value,parts=ddp(*transferred,coefficients,n,k);(value*8192).backward()
        totals+=parts
    finish_gradients(model,scale=8192,clip=float('inf'));dist.all_reduce(totals)
    gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters()}
    if any(not torch.isfinite(g).all() for g in gradients.values()):raise FloatingPointError('Nonfinite mature objective gradient')
    after=digest(dict(sorted(model.state_dict().items())))
    if before!=after:raise AssertionError('Zero-update probe changed weights')
    noise_file=a.output/f'h0-rank{rank}.npy';np.save(noise_file,torch.cat(noise).numpy())
    replica={'model_digest':after,'gradient_digest':digest(gradients),'rows':rows,
             'noise_sha256':file_sha256(noise_file),'fallback_ops':list(torch_neuronx.get_fallback_ops())}
    (a.output/f'rank{rank}.json').write_text(json.dumps(replica)+'\n');dist.barrier()
    if rank==0:
        replicas=[json.loads((a.output/f'rank{i}.json').read_text()) for i in range(32)]
        if len({r['model_digest'] for r in replicas})!=1 or len({r['gradient_digest'] for r in replicas})!=1:
            raise AssertionError('Complete model/gradient replicas differ')
        if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in replicas):raise AssertionError('Unexpected probe fallback')
        rows=sorted([r for replica in replicas for r in replica['rows']],key=lambda r:r['window'])
        if [r['window'] for r in rows]!=list(range(cursor,cursor+512)) or sum(r['valid_anchors'] for r in rows)!=anchor_count:
            raise AssertionError('Complete source/anchor objective coverage differs')
        vector=a.output/'gradient.pt';atomic_checkpoint(vector,{'gradients':gradients,
            'ntp_nll':float(totals[0].cpu())/total,'future_mean':float(totals[1].cpu())/max(1,anchor_count)})
        matrix=sum(matmul_ledger(config,1024,total_steps=r['n'],backward_cap=r['k'])['training_matmul_flops'] for r in rows)
        matrix+=512*6*len(span.positions)*config.dim*config.vocab_size
        report={'status':'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT','case_id':a.case,'cell':case['cell'],
            'plan_sha256':a.plan_sha,'checkpoint_sha256':a.checkpoint_sha,'full_state_digest':complete,
            'training_update':update,'source_cursor':cursor,'real_targets':total,'valid_anchors':anchor_count,
            'windows':512,'rows':rows,'gradient_sha256':file_sha256(vector),'model_unchanged':True,
            'noise_sha256_by_rank':[r['noise_sha256'] for r in replicas],'replica_gradients_equal':True,
            'diagnostic_matrix_ops':matrix,'optimizer_updates':0,'research_optimizer_updates':0,
            'test_accessed':False,'wall_seconds':time.time()-started,
            'scope':'Complete next512-window natural TRAIN objective at a registered fitted checkpoint. No clipping,optimizer update,cursor advancement or quality result. Independent FP32 reference gate required.'}
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
