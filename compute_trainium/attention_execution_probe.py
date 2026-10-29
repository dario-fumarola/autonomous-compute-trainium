"""Numerical operator fixtures, never research examples or model training."""
import math
import torch


def attention(q,k,v,tile=None):
    n=q.shape[-2];scale=math.sqrt(q.shape[-1])
    if tile is None:
        scores=q@k.transpose(-1,-2)/scale
        mask=torch.ones((n,n),dtype=torch.bool,device=q.device).tril()
        return scores.masked_fill(~mask,float('-inf')).softmax(-1)@v
    columns=torch.arange(n,device=q.device)
    result=[]
    for start in range(0,n,tile):
        end=min(start+tile,n)
        scores=q[...,start:end,:]@k.transpose(-1,-2)/scale
        allowed=columns[None,:]<=torch.arange(start,end,device=q.device)[:,None]
        result.append(scores.masked_fill(~allowed,float('-inf')).softmax(-1)@v)
    return torch.cat(result,-2)


CASES=(('attention512',512,None),('attention1024',1024,None),
       ('tiled128',1024,128),('tiled256',1024,256),
       ('softmax512',512,None),('softmax1024',1024,None),
       ('qk1024',1024,None),('pv1024',1024,None))


def make_inputs(case):
    label,n,tile=CASES[case]
    generator=torch.Generator().manual_seed(9841+case)
    if label.startswith('softmax'):
        return [torch.randn(2,8,n,n,generator=generator)]
    if label=='pv1024':
        return [torch.randn(2,8,n,n,generator=generator).softmax(-1),
                torch.randn(2,8,n,64,generator=generator)]
    count=2 if label=='qk1024' else 3
    return [torch.randn(2,8,n,64,generator=generator) for _ in range(count)]


def operation(case,*args):
    label,n,tile=CASES[case]
    if label.startswith('softmax'):
        mask=torch.ones((n,n),device=args[0].device,dtype=torch.bool).tril()
        return args[0].masked_fill(~mask,float('-inf')).softmax(-1)
    if label=='qk1024':return args[0]@args[1].transpose(-1,-2)/8
    if label=='pv1024':return args[0]@args[1]
    return attention(*args,tile=tile)


def objective(case,*args):
    return operation(case,*args).square().mean()
