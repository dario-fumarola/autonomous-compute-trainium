"""Capture an actual global gradient, then diagnose it against CPU precisions.

FP32/BF16 partition controls. No optimizer updates, no relaxed training guard, no TEST. Capture and CPU
reference are separate processes so accelerator ranks can exit before the
large independent CPU accumulation. Use the same native SDK for both.
"""
import argparse
import contextlib
import copy
import json
import os
import sys
import time
from pathlib import Path

if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1]=='capture':
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

import torch

from compute_trainium.depth_mixture import routed_indices
from compute_trainium.preln_recurrence import PreNormRecurrenceConfig,build_preln_recurrent
from compute_trainium.training import PackedCorpus,TokenObjective,atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def compare_gradients(left,right):
    if left.keys()!=right.keys():raise ValueError('Gradient parameter sets differ')
    error=norm=other=dot=0.;maximum=0.;rows=[]
    for name in left:
        a,b=left[name].double(),right[name].double()
        if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Invalid gradient tensors')
        delta=float((a-b).square().sum());scale=float(b.square().sum());error+=delta;norm+=scale
        other+=float(a.square().sum());dot+=float((a*b).sum());maximum=max(maximum,float((a-b).abs().max()))
        rows.append({'parameter':name,'squared_error':delta,'reference_squared_norm':scale})
    return {'relative_l2':(error/max(norm,1e-30))**.5,'cosine':dot/max((norm*other)**.5,1e-30),
            'reference_norm':norm**.5,'candidate_norm':other**.5,'maximum_absolute_error':maximum,
            'largest_error_contributors':sorted(rows,key=lambda x:-x['squared_error'])[:12]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--packed-corpus',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['capture','reference'],required=True)
    p.add_argument('--capture',type=Path);p.add_argument('--native-precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--capture-batch',type=int,choices=[1,4],default=4);p.add_argument('--capture-accumulation',type=int,choices=[2,8],default=2)
    p.add_argument('--deadline-unix',type=float,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True);started=time.time();torch.set_num_threads(2 if a.mode=='capture' else 48)
    saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False);sig=saved['signature']
    if sig['world']!=32 or sig['batch']!=4 or sig['accumulation']!=2:
        raise ValueError('This diagnostic requires the preserved32x4x2 source topology')
    if saved['pauses'] or file_sha256(a.corpus)!=sig['corpus_sha256']:raise ValueError('Unexpected data or pause path')
    config=PreNormRecurrenceConfig(**sig['model']);model=build_preln_recurrent(config);model.load_state_dict(saved['model'])
    packed=PackedCorpus.from_mapped(a.packed_corpus,1024,sig['data_seed'],sig['corpus_sha256']);cursor=saved['cursor']
    if packed.fingerprint!=sig['packing_sha256']:raise AssertionError('Packing differs from checkpoint')
    count=int((packed.prefix_counts[cursor+256]-packed.prefix_counts[cursor]).sum())
    size,accumulation=(a.capture_batch,a.capture_accumulation) if a.mode=='capture' else (4,2)
    if size*accumulation!=8:raise ValueError('Preserve all256 global source windows')
    def batch(worker,micro):return packed.batch(routed_indices(cursor,micro,worker,32,size,accumulation,sig['routing_seed']),0)[0]
    first=batch(0,0);checkpoint_sha=file_sha256(a.checkpoint)
    def gradients(module):return {name:parameter.grad.detach().cpu().clone() for name,parameter in module.named_parameters()}
    if a.mode=='capture':
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        import torch_neuronx
        cores=os.environ['RESEARCH_VISIBLE_CORES'].split(',');rank=int(os.environ['LOCAL_RANK'])
        os.environ['NEURON_RT_VISIBLE_CORES']=cores[rank]
        torch.neuron.set_device(0);dist.init_process_group('neuron')
        model.to('neuron');own=batch(rank,0)
        objective=TokenObjective(model,first.prediction_positions,a.native_precision=='bf16')
        torch._dynamo.config.optimize_ddp=False
        compiled=torch.compile(objective,backend='neuron',fullgraph=True,dynamic=False)
        local=compiled(own.inputs.contiguous().to('neuron'),own.targets.contiguous().to('neuron'),own.target_mask.contiguous().to('neuron'))/own.target_mask.sum().item()
        local.backward()
        if rank==0:atomic_checkpoint(a.output/'local-native.pt',{'gradients':gradients(model),'loss':float(local.detach().cpu())})
        del local;model.zero_grad(set_to_none=True)
        ddp=DDP(compiled,device_ids=None,broadcast_buffers=False,bucket_cap_mb=64,gradient_as_bucket_view=True)
        loss_sum=torch.zeros((),device='neuron')
        for micro in range(accumulation):
            own=batch(rank,micro)
            with ddp.no_sync() if micro<accumulation-1 else contextlib.nullcontext():
                loss=ddp(own.inputs.contiguous().to('neuron'),own.targets.contiguous().to('neuron'),own.target_mask.contiguous().to('neuron'))
                (loss*(32/count)).backward();loss_sum+=loss.detach()
        dist.all_reduce(loss_sum);native_gradients=gradients(model)
        gradient_digest=digest(native_gradients)
        (a.output/f'replica-{rank}.json').write_text(json.dumps({'gradient_digest':gradient_digest})+'\n')
        dist.barrier()
        if rank==0:
            if len({json.loads((a.output/f'replica-{r}.json').read_text())['gradient_digest'] for r in range(32)})!=1:raise AssertionError('Replica gradients differ')
            if set(torch_neuronx.get_fallback_ops())-{'aten::_to_copy'}:raise AssertionError('Unexpected native fallback')
            atomic_checkpoint(a.output/'global-native.pt',{'gradients':native_gradients,'loss':float(loss_sum.cpu())/count})
            report={'status':'CAPTURED_NATIVE_GLOBAL_GRADIENT_NO_UPDATES','checkpoint_sha256':checkpoint_sha,
                'real_targets':count,'source_update':saved['update'],'all_replica_gradients_identical':True,
                'native_precision':a.native_precision,'capture_batch':size,'capture_accumulation':accumulation,
                'gradient_digest':gradient_digest,'native_global_loss':float(loss_sum.cpu())/count,'wall_seconds':time.time()-started}
            (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        dist.barrier();dist.destroy_process_group();return
    if a.capture is None:raise ValueError('Reference requires a completed capture')
    capture=json.loads((a.capture/'report.json').read_text())
    if capture['checkpoint_sha256']!=checkpoint_sha or capture['real_targets']!=count:raise AssertionError('Capture source differs')
    native=torch.load(a.capture/'global-native.pt',map_location='cpu',mmap=True,weights_only=False)
    local_native=torch.load(a.capture/'local-native.pt',map_location='cpu',mmap=True,weights_only=False)
    comparisons={};losses={}
    for bf16 in [False,True]:
        label='bf16' if bf16 else 'fp32';model.zero_grad(set_to_none=True)
        objective=TokenObjective(model,first.prediction_positions,bf16,'cpu')
        local_loss=objective(first.inputs,first.targets,first.target_mask)/first.target_mask.sum()
        local_loss.backward();comparisons['local_native_vs_cpu_'+label]=compare_gradients(local_native['gradients'],gradients(model))
        model.zero_grad(set_to_none=True);total=0.;targets=0
        for micro in range(2):
            for worker in range(32):
                own=batch(worker,micro)
                for row in range(4):
                    loss=objective(own.inputs[row:row+1],own.targets[row:row+1],own.target_mask[row:row+1])
                    (loss/count).backward();total+=float(loss.detach());targets+=int(own.target_mask[row:row+1].sum())
                if time.time()>a.deadline_unix:raise TimeoutError('CPU global reference deadline')
            print(json.dumps({'event':'cpu_global_microbatch_complete','precision':label,'micro':micro,'seconds':time.time()-started}),flush=True)
        if targets!=count:raise AssertionError('Global objective target counts differ')
        current=gradients(model);comparisons['global_native_vs_cpu_'+label]=compare_gradients(native['gradients'],current)
        atomic_checkpoint(a.output/('global-cpu-'+label+'.pt'),{'gradients':current,'loss':total/count});losses[label]=total/count
        if bf16:
            full=torch.load(a.output/'global-cpu-fp32.pt',map_location='cpu',mmap=True,weights_only=False)
            comparisons['global_cpu_bf16_vs_cpu_fp32']=compare_gradients(current,full['gradients'])
    if file_sha256(a.checkpoint)!=checkpoint_sha:raise AssertionError('Source checkpoint changed')
    qualifies=comparisons['global_native_vs_cpu_fp32']['relative_l2']<=.02 and abs(native['loss']-losses['fp32'])<=.003
    report={'status':'COMPLETE_GLOBAL_PRECISION_DIAGNOSIS','qualifies_global_fp32_gradient_gate':qualifies,
        'checkpoint_sha256':checkpoint_sha,'source_update':saved['update'],'real_targets':count,
        'native_loss':native['loss'],'cpu_losses':losses,'comparisons':comparisons,'wall_seconds':time.time()-started,
        'scope':'No optimizer updates or TEST. Global real-token objective compared with full FP32 CPU accumulation, plus BF16 and exact local probes. Qualification thresholds predeclared: global gradient2%, loss0.003. This does not silently modify or relabel the failed8% local training gate.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
