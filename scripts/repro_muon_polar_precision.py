"""Bounded Muon arithmetic diagnosis; numerical arrays, never research data."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA'] = '0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES'] = os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK', '0'))]

import torch
import torch.distributed as dist

from compute_trainium.parcae_optimizer_candidate import POLAR_COEFFS


def polar_trace(gradient, *, polar_dtype=torch.bfloat16):
    # First-step Nesterov momentum; same operations as the released optimizer.
    momentum = torch.tensor(.85, dtype=gradient.dtype, device=gradient.device)
    m = torch.zeros_like(gradient).lerp_(gradient, 1-momentum)
    update = gradient.clone().lerp_(m, momentum)
    x = update.to(polar_dtype)
    denominator = x.norm(dim=(-2, -1), keepdim=True)*1.02+1e-6
    x = x/denominator
    outputs = [m, update, denominator, x]
    for a, b, c in POLAR_COEFFS:
        if gradient.size(-2)>gradient.size(-1):
            gram = x.mT@x
            x = a*x+x@(b*gram+c*(gram@gram))
        else:
            gram = x@x.mT
            x = a*x+(b*gram+c*(gram@gram))@x
        outputs.extend([gram, x])
    return tuple(outputs)


def errors(expected, actual):
    if len(expected)!=len(actual):raise ValueError('Stage inventory differs')
    rows=[]
    for i,(a,b) in enumerate(zip(expected,actual,strict=True)):
        a,b=a.float().double(),b.float().double()
        rows.append({'stage':i,'shape':list(a.shape),'relative_l2':float((a-b).norm()/a.norm().clamp_min(1e-30)),
                     'maximum_absolute_error':float((a-b).abs().max()),'reference_norm':float(a.norm())})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--corpus');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=8:raise ValueError('Eight independent numerical cases, one core per chip')
    shapes=[(1,768,768),(1,3072,768),(1,768,3072),(1,6,32)]
    shape=shapes[rank%4];dtype=torch.bfloat16 if rank<4 else torch.float32
    generator=torch.Generator().manual_seed(739+rank%4)
    gradient=torch.randn(shape,generator=generator)*.03
    started=time.time()
    expected=polar_trace(gradient,polar_dtype=dtype)
    native=gradient.to('neuron')
    eager=tuple(x.cpu() for x in polar_trace(native,polar_dtype=dtype))
    compiled=torch.compile(lambda x:polar_trace(x,polar_dtype=dtype),backend='neuron',fullgraph=True,dynamic=False)
    actual=tuple(x.cpu() for x in compiled(native))
    row={'rank':rank,'shape':shape,'polar_dtype':str(dtype),'cpu_eager_vs_native_eager':errors(expected,eager),
         'cpu_eager_vs_native_compiled':errors(expected,actual),'native_eager_vs_native_compiled':errors(eager,actual),
         'wall_seconds':time.time()-started,'fallback_ops':sorted(torch_neuronx.get_fallback_ops())}
    (args.output/f'rank-{rank}.json').write_text(json.dumps(row,indent=2)+'\n')
    dist.barrier()
    if rank==0:
        rows=[json.loads((args.output/f'rank-{i}.json').read_text()) for i in range(world)]
        report={'status':'COMPLETE_MUON_POLAR_ARITHMETIC_DIAGNOSIS','records':rows,'torch':torch.__version__,
                'optimizer_updates':0,'research_optimizer_updates':0,'test_accessed':False,
                'scope':'Same-version CPU eager/native eager/compiled intermediate arithmetic, numerical random-array fixtures only. FP32 is a diagnostic, not an adopted optimizer recipe.'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status']}),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
