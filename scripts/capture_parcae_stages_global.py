"""Native static-stage FP32 gradient capture, zero optimizer updates."""
import argparse
import contextlib
import ctypes
import gc
import json
import math
import os
import time
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_stages_candidate import ParcaeStagesCandidate
from compute_trainium.depth_mixture import routed_indices
from compute_trainium.metrics import real_token_losses
from compute_trainium.training import PackedCorpus, atomic_checkpoint, file_sha256
from scripts.qualify_native import digest


def canonical(state):
    return digest(dict(sorted(state.items())))


def loss_sum(logits,targets,mask):
    return real_token_losses(logits,targets,mask).sum()


class Objective(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
        self.loss=torch.compile(loss_sum,backend='neuron',fullgraph=True,dynamic=False)

    def forward(self,inputs,targets,mask,state=None):
        logits=self.model(inputs,state,total_steps=4 if self.model.recurrent else 1,
                          backward_cap=2 if self.model.recurrent else 1)
        return self.loss(logits,targets,mask)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--packed-corpus',type=Path,required=True)
    p.add_argument('--arm',choices=['ordinary','recurrent'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch_neuronx
    torch.set_num_threads(2)
    torch.neuron.set_device(0)
    dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=32:
        raise ValueError('Registered32-rank half-node capture required')
    args.output.mkdir(exist_ok=True)
    started=time.time()
    plan=json.loads((args.reference_root/'plan.json').read_text())
    proof=json.loads((args.reference_root/'report.json').read_text())
    if proof['status']!='COMPLETE_INDEPENDENT_PARCAE_CPU_GLOBAL_REFERENCES':
        raise ValueError('Both independent global objectives must be complete')
    reference=next(r for r in proof['records'] if r['arm']==args.arm)
    checkpoint=args.reference_root/args.arm/'qualification-state.pt'
    if rank==0:
        if file_sha256(checkpoint)!=reference['checkpoint_sha256']:
            raise AssertionError('Qualification weights changed')
        if file_sha256(args.reference_root/'initial-state.npy')!=plan['initial_state_file_sha256']:
            raise AssertionError('Saved initial-state fixtures changed')
    dist.barrier()
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if not saved['qualification_only'] or saved['recurrent']!=(args.arm=='recurrent'):
        raise AssertionError('Qualification state identity mismatch')
    model=ParcaeStagesCandidate(ParcaeCandidateConfig(**saved['config']),
        recurrent=saved['recurrent'],initialize=False)
    expected_state=canonical(saved['model'])
    model.load_state_dict(saved['model'],strict=True)
    if canonical(model.state_dict())!=expected_state:
        raise AssertionError('Loading reference weights changed the model')
    del saved
    if file_sha256(args.corpus)!=plan['source_corpus_sha256']:
        raise AssertionError('Qualification source changed')
    packed=PackedCorpus.from_mapped(args.packed_corpus,1024,23,plan['source_corpus_sha256'])
    if packed.fingerprint!=plan['packing_fingerprint'] or sum(packed.counts(0,256))!=plan['real_targets']:
        raise AssertionError('Qualification packing or real-target count changed')
    noise=np.load(args.reference_root/'initial-state.npy',mmap_mode='r')
    def batch(micro):
        index=routed_indices(0,micro,rank,32,1,8,271828)[0]
        item=packed.batch([index],0)[0]
        state=torch.from_numpy(np.array(noise[index:index+1],copy=True)) if model.recurrent else None
        values=[v.contiguous().to('neuron') for v in (item.inputs,item.targets,item.target_mask)]
        return item,(*values,state.contiguous().to('neuron') if state is not None else None)
    model.to('neuron')
    if any(p.dtype!=torch.float32 for p in model.parameters()):
        raise AssertionError('Expected FP32 master parameters')
    gc.collect();ctypes.CDLL(None).malloc_trim(0)
    torch._dynamo.config.optimize_ddp=False
    model.compile_stages('neuron')
    objective=Objective(model)
    own,values=batch(0)
    local=objective(*values)/int(own.target_mask.sum())
    local.backward()
    if rank==0:
        gradients={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters()}
        cpu=torch.load(args.reference_root/args.arm/'local-cpu-fp32.pt',map_location='cpu',mmap=True,weights_only=False)
        error=norm=0.
        if set(gradients)!=set(cpu['gradients']):
            raise AssertionError('Local gradient parameter names differ')
        for name,value in gradients.items():
            error+=float((value-cpu['gradients'][name]).double().square().sum())
            norm+=float(cpu['gradients'][name].double().square().sum())
        relative=(error/max(norm,1e-30))**.5
        native_loss=float(local.detach().cpu())
        if not math.isfinite(relative) or not math.isfinite(native_loss) or relative>.02 or abs(native_loss-cpu['loss'])>.003:
            raise AssertionError('Local native/independent-CPU probe failed')
        atomic_checkpoint(args.output/'local-native.pt',{'gradients':gradients,'loss':float(local.detach().cpu())})
        del gradients,cpu
    model.zero_grad(set_to_none=True)
    ddp=DDP(objective,device_ids=None,broadcast_buffers=False,bucket_cap_mb=64,gradient_as_bucket_view=True)
    total=torch.zeros((),device='neuron')
    for micro in range(8):
        _,values=batch(micro)
        with ddp.no_sync() if micro<7 else contextlib.nullcontext():
            value=ddp(*values)
            (value*(32/plan['real_targets'])).backward()
        total+=value.detach()
    dist.all_reduce(total)
    gradients={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters()}
    if not all(g.dtype==torch.float32 and torch.isfinite(g).all() for g in gradients.values()):
        raise AssertionError('Nonfinite or non-FP32 native gradient')
    after=canonical(model.state_dict())
    if after!=expected_state:
        raise AssertionError('Gradient capture changed model weights')
    gradient_digest=digest(dict(sorted(gradients.items())))
    (args.output/f'replica-{rank}.json').write_text(json.dumps({'gradient_digest':gradient_digest,
        'weight_digest':after,'fallback_ops':list(torch_neuronx.get_fallback_ops())})+'\n')
    dist.barrier()
    if rank==0:
        replicas=[json.loads((args.output/f'replica-{i}.json').read_text()) for i in range(32)]
        if len({r['gradient_digest'] for r in replicas})!=1 or len({r['weight_digest'] for r in replicas})!=1:
            raise AssertionError('Native replicas disagree')
        if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in replicas):
            raise AssertionError('Unexpected native fallback')
        atomic_checkpoint(args.output/'global-native.pt',{'gradients':gradients,'loss':float(total.cpu())/plan['real_targets']})
        if file_sha256(checkpoint)!=reference['checkpoint_sha256']:
            raise AssertionError('Qualification checkpoint file changed')
        report={'status':'CAPTURED_NATIVE_PARCAE_STATIC_STAGE_FP32_GLOBAL_GRADIENT',
            'arm':args.arm,'parameters':model.config.parameter_count(model.recurrent),
            'real_targets':plan['real_targets'],'native_nll':float(total.cpu())/plan['real_targets'],
            'weights_unchanged':True,'all_replica_gradients_equal':True,'optimizer_updates':0,
            'test_accessed':False,'wall_seconds':time.time()-started,
            'scope':'Native compiled encode/refine/decode plus CE, same256 natural TRAIN windows/initial states as independent upstream CPU. Full objective comparison is performed separately; no optimizer, recovery, sampled-depth or throughput qualification.'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
