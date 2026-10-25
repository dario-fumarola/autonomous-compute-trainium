"""CPU-only SDK adjoint reproduction;needs PyTorch and one installed Python file.

No project imports,private data,model checkpoints,accelerators or research
examples. Run with --sdk-source /path/to/torch_neuronx/_sdpa_math.py.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sdk-source',type=Path,required=True)
    parser.add_argument('--sequence-length',type=int,default=32)
    args=parser.parse_args()
    if args.sequence_length not in (32,1024):raise ValueError('Use either original frozen fixture length')
    torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('isolated_math2_source',args.sdk_source)
    sdk=importlib.util.module_from_spec(spec);spec.loader.exec_module(sdk)
    generator=torch.Generator().manual_seed(109)
    query=F.rms_norm(torch.randn(1,6,args.sequence_length,128,generator=generator),(128,)).half()
    key=query.clone();value=(.5*torch.randn(query.shape,generator=generator)).half()
    grad_out=torch.randn(query.shape,generator=generator).half()
    query,key,value=[tensor.requires_grad_() for tensor in (query,key,value)]
    scale=1/math.sqrt(128)
    output,lse=sdk.compute_attention(query,key,value,None,None,scale,True,0.)
    output.backward(grad_out)
    expected=[tensor.grad.detach() for tensor in (query,key,value)]
    actual=sdk.compute_attention_backward(grad_out,query.detach(),key.detach(),value.detach(),
        None,None,output.detach(),lse.detach(),scale,True,0.)[:3]
    errors={name:float((truth.double()-observed.double()).norm()/truth.double().norm().clamp_min(1e-30))
            for name,truth,observed in zip(('query','key','value'),expected,actual)}
    print(json.dumps({'status':'REPRODUCED_BACKWARD_DISCREPANCY' if max(errors.values())>.02 else 'NO_LARGE_DISCREPANCY',
        'sdk_source_sha256':hashlib.sha256(args.sdk_source.read_bytes()).hexdigest(),
        'torch_version':torch.__version__,'sequence_length':args.sequence_length,
        'relative_l2_errors':errors,'reference':'PyTorch autograd through the identical SDK forward',
        'optimizer_updates':0,'research_examples':0,'native_chips':0},indent=2))


if __name__=='__main__':main()
