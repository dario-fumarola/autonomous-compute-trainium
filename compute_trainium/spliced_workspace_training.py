"""Ordinary next-token loss,with sampled independent workspace predictions.

Each authentic target appears ONCE.No auxiliary target weighting,PG or teacher.
Sampling depends only on an explicit RNG,not source contents or outcomes.
"""
import torch
from .stable_target_loss import stable_target_log_probs


def partition_positions(anchors,*,target_count=159):
    if anchors.device.type!='cpu' or anchors.dtype!=torch.long or anchors.ndim!=1:
        raise ValueError('Prepare a CPU int64 query schedule before Native execution')
    if anchors.numel()==0 or not bool(((anchors>=0)&(anchors<target_count)).all()) or torch.unique(anchors).numel()!=anchors.numel():
        raise ValueError('Distinct in-range query positions required')
    selected=anchors.sort().values;keep=torch.ones(target_count,dtype=torch.bool);keep[selected]=False
    return selected,torch.arange(target_count)[keep]


def sample_positions(generator,*,count=32,target_count=159,eligible_start=31):
    if type(count) is not int or not 1<=count<=target_count-eligible_start or eligible_start<0:
        raise ValueError('Bounded fixed-count prefix-independent opportunities required')
    selected=torch.randperm(target_count-eligible_start,generator=generator)[:count]+eligible_start
    return partition_positions(selected,target_count=target_count)


def spliced_objective(model,tokens,targets,cos,sin,anchors,raw_positions,*,mode='latent',pause_id=None):
    if mode=='ordinary':
        logits=model(tokens,cos,sin,positions=torch.arange(159,device=tokens.device))
        losses=-stable_target_log_probs(logits,targets)
        return losses.mean(),losses.index_select(1,raw_positions).mean(),losses.index_select(1,anchors).mean()
    if mode not in ['latent','constant']:raise ValueError('Registered workspace mode required')
    r=model.sparse_forward(tokens,cos,sin,anchors,raw_positions=raw_positions,mode=mode,pause_id=pause_id,mix=.5)
    raw=-stable_target_log_probs(r['raw_logits'],targets.index_select(1,raw_positions))
    work=-stable_target_log_probs(r['mixed_logits'],targets.index_select(1,anchors))
    loss=(raw.sum()+work.sum())/targets.numel()
    return loss,raw.mean(),work.mean()


def spliced_matrix_work(config,*,query_count=32):
    if type(query_count) is not int or not 0<=query_count<=159:raise ValueError('Bounded query count required')
    d,f,l,h,kv=config.hidden_size,config.intermediate_size,config.num_hidden_layers,config.num_attention_heads,config.num_key_value_heads
    kv_width=d*kv//h;n=160;q=query_count
    dense=2*(2*d*d+2*d*kv_width+3*d*f)*(n+q)*l
    attention=4*d*(n*n+q*(n+q))*l
    # Every natural target is projected once,regardless of its compute path.
    return 4*(dense+attention)+3*2*159*d*config.vocab_size
