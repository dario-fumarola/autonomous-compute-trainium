"""Zero-update complete-global-objective capture at registered checkpoints."""
import argparse
import contextlib
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
from compute_trainium.low_depth_gradient_probe import validate_probe_state
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_fp16_candidate import ParcaeFP16Candidate
from compute_trainium.parcae_stages_candidate import ParcaeStagesCandidate
from compute_trainium.parcae_training_candidate import initial_state,map_after_shared_validation
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--point',required=True);p.add_argument('--comm-port',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    a=p.parse_args();started=time.time()
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Gradient-probe plan changed')
    probe=json.loads(a.plan.read_text());point=probe['points'][a.point]
    if probe['purpose']!='zero_update_low_depth_gradient_validation':raise ValueError('Finite gradient-only plan required')
    if probe['root_comm_ports'][a.output.name]!=a.comm_port:raise ValueError('Unregistered probe communication port')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    from scripts.train_parcae_precise_qualification import Objective
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=32:raise ValueError('Complete32-rank global probe required')
    code_root=Path(__file__).resolve().parents[1]
    if any(file_sha256(code_root/f)!=sha for f,sha in probe['worker_source_sha256'].items()):
        raise AssertionError('Frozen probe implementation changed')
    parent_file=a.root/probe['parent_plan']
    if file_sha256(parent_file)!=probe['parent_plan_sha256']:raise AssertionError('Parent scientific plan changed')
    parent=json.loads(parent_file.read_text());config=ParcaeCandidateConfig(**parent['model'])
    checkpoint=a.root/point['checkpoint']
    if rank==0 and file_sha256(checkpoint)!=point['checkpoint_sha256']:raise AssertionError('Probe checkpoint changed')
    dist.barrier();saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    case=validate_probe_state(saved,probe['parent_plan_sha256'],parent,point['case_id'])
    full_digest=digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})
    if full_digest!=saved['model_optimizer_digest']:raise AssertionError('Probe full-state readback differs')
    cursor,update=saved['cursor'],saved['update'];recurrent=case['arm']!='ordinary'
    cls=ParcaeFP16Candidate if parent['precision']=='fp16' else ParcaeStagesCandidate
    model=cls(config,recurrent=recurrent,initialize=False);model.load_state_dict(saved['model'])
    before=digest(dict(sorted(model.state_dict().items())));del saved
    packed_path=a.root/parent['data']['packed']
    if rank==0:
        if file_sha256(a.root/parent['data']['corpus'])!=parent['data']['corpus_sha256']:
            raise AssertionError('Natural TRAIN source changed')
        checked=PackedCorpus.from_mapped(packed_path,1024,23,parent['data']['corpus_sha256'])
        if checked.fingerprint!=parent['data']['packing_fingerprint']:raise AssertionError('Natural packing changed')
        del checked
    dist.barrier();packed=map_after_shared_validation(packed_path,parent['data']['manifest_sha256'])
    depths_file=parent_file.parent/case['depth_table']
    if rank==0 and file_sha256(depths_file)!=case['depth_table_sha256']:raise AssertionError('Probe depth table changed')
    dist.barrier();depths=np.load(depths_file,mmap_mode='r')
    accumulation=parent['accumulation'];windows=32*accumulation
    if update>=len(depths) or cursor+windows>packed.windows:raise ValueError('Provisioned next full TRAIN objective required')
    total=sum(packed.counts(cursor,windows));model.to('neuron');torch._dynamo.config.optimize_ddp=False
    model.compile_stages('neuron');ddp=DDP(Objective(model),device_ids=None,broadcast_buffers=False,
        bucket_cap_mb=64,gradient_as_bucket_view=True)
    scale=8192 if parent['precision']=='fp16' else 1
    a.output.mkdir(exist_ok=True)
    loss=torch.zeros((),device='neuron');local_rows=[];local_states=[]
    for micro in range(accumulation):
        window=routed_indices(cursor,micro,rank,32,1,accumulation,parent['routing_seed'])[0]
        item=packed.batch([window],0)[0];n,k=map(int,depths[update,micro,rank])
        h0=initial_state(config,window,case['state_seed']) if recurrent else None
        local_rows.append({'window':window,'rank':rank,'micro':micro,'n':n,'k':k,
                           'h0_digest':digest(h0) if recurrent else None})
        if recurrent:local_states.append(h0)
        inputs,targets,mask=[t.contiguous().to('neuron') for t in [item.inputs,item.targets,item.target_mask]]
        with ddp.no_sync() if micro<accumulation-1 else contextlib.nullcontext():
            value=ddp(inputs,targets,mask,h0.to('neuron') if recurrent else None,n,k)
            (value*(world/total)*scale).backward()
        loss+=value.detach()
    dist.all_reduce(loss)
    for parameter in model.parameters():
        if parameter.grad is None or parameter.grad.dtype!=torch.float32:raise AssertionError('Missing full master gradient')
        parameter.grad.div_(scale)
    gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters()}
    if any(not torch.isfinite(g).all() for g in gradients.values()):raise FloatingPointError('Nonfinite raw probe gradient')
    after=digest(dict(sorted(model.state_dict().items())))
    if before!=after:raise AssertionError('Zero-update probe changed weights')
    noise_file=a.output/f'probe-h0-rank{rank}.npy'
    if recurrent:np.save(noise_file,torch.cat(local_states).numpy())
    record={'model_digest':after,'gradient_digest':digest(gradients),'probe_rows':local_rows,
            'noise_sha256':file_sha256(noise_file) if recurrent else None,
            'fallback_ops':list(torch_neuronx.get_fallback_ops())}
    (a.output/f'rank{rank}.json').write_text(json.dumps(record)+'\n');dist.barrier()
    if rank==0:
        replicas=[json.loads((a.output/f'rank{i}.json').read_text()) for i in range(32)]
        if len({r['model_digest'] for r in replicas})!=1 or len({r['gradient_digest'] for r in replicas})!=1:
            raise AssertionError('Gradient probe replicas differ')
        if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in replicas):raise AssertionError('Unexpected gradient-probe fallback')
        rows=sorted([row for r in replicas for row in r['probe_rows']],key=lambda row:row['window'])
        if [r['window'] for r in rows]!=list(range(cursor,cursor+windows)):raise AssertionError('Global probe source coverage differs')
        vector=a.output/'gradient.pt';atomic_checkpoint(vector,{'gradients':gradients,'loss':float(loss.cpu())/total})
        work=sum(matmul_ledger(config,1024,1,recurrent=recurrent,total_steps=r['n'],backward_cap=r['k'])['training_matmul_flops'] for r in rows)
        result={'status':'CAPTURED_ZERO_UPDATE_LOW_DEPTH_GLOBAL_GRADIENT','point':a.point,
            'case_id':point['case_id'],'arm':case['arm'],'checkpoint_sha256':point['checkpoint_sha256'],
            'parent_plan_sha256':probe['parent_plan_sha256'],'probe_plan_sha256':a.plan_sha,
            'source_cursor':cursor,'training_update':update,'real_targets':total,'windows':windows,
            'gradient_sha256':file_sha256(vector),'full_state_digest':full_digest,'model_unchanged':True,
            'replica_gradients_equal':True,'probe_rows':rows,'neural_matmuls':work,
            'noise_sha256_by_rank':[r['noise_sha256'] for r in replicas],
            'optimizer_updates':0,'research_optimizer_updates':0,'test_accessed':False,
            'wall_seconds':time.time()-started,'scope':'Next full TRAIN objective at frozen checkpoint. No clipping, optimizer step, source advancement, held-out selection or precision claim without independent CPU gradient comparison.'}
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
