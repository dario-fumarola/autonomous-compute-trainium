"""Trainable,efficient form of the registered independent-memory intervention."""
from collections import Counter
import hashlib
import torch
from torch.nn import functional as F
from .residual_workspace import unit


def independent_read(model,features,anchors,donor):
    if model.mode!='one' or donor.shape!=features.shape:raise ValueError('One supported step and matched memory geometry required')
    base=features.index_select(1,anchors);state=model.input(unit(base));memory=model.input(unit(donor));cell=model.cell
    query=cell.q(unit(state));key=cell.k(unit(memory));value=cell.v(unit(memory));current_key=cell.k(unit(state));current_value=cell.v(unit(state));scale=state.shape[-1]**-.5
    scores=(query@key.transpose(-1,-2))*scale;index=anchors[None,:,None].expand(features.shape[0],-1,1);own_score=(query*current_key).sum(-1,keepdim=True)*scale;scores=scores.scatter(-1,index,own_score)
    mask=torch.arange(features.shape[1],device=features.device)[None,:]<=anchors[:,None];weights=scores.masked_fill(~mask,float('-inf')).softmax(-1)
    # Linear attention aggregation lets the original self value replace the
    # donor self value without materializing a separate full memory per query.
    read=weights@value+weights.gather(-1,index)*(current_value-value.index_select(1,anchors))
    state=state+cell.o(read);state=state+cell.down(F.silu(cell.up(unit(state))))
    return base+model.output(unit(state))


def source_group_derangement(domains,groups):
    if len(domains)!=len(groups):raise ValueError('Exact source metadata required')
    result=[None]*len(groups)
    for domain in sorted(set(domains)):
        indices=sorted([i for i,d in enumerate(domains) if d==domain],key=lambda i:(groups[i],i));largest=max(Counter(groups[i] for i in indices).values())
        if 2*largest>len(indices):raise ValueError('A source group dominates the domain;no group-disjoint permutation exists')
        for k,i in enumerate(indices):result[i]=indices[(k+largest)%len(indices)]
    if sorted(result)!=list(range(len(groups))) or any(domains[i]!=domains[j] or groups[i]==groups[j] for i,j in enumerate(result)):raise AssertionError('Exact same-domain independent permutation failed')
    return result


def fixed_evaluation_donors(train_domains,train_groups,eval_domains,eval_groups,*,seed=7297):
    pools={d:[i for i,x in enumerate(train_domains) if x==d] for d in set(train_domains)};result=[]
    for i,(domain,group) in enumerate(zip(eval_domains,eval_groups,strict=True)):
        pool=pools[domain];start=int(hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest(),16)%len(pool);chosen=[];used={group}
        for step in range(len(pool)):
            j=pool[(start+step)%len(pool)]
            if train_groups[j] in used:continue
            chosen.append(j);used.add(train_groups[j])
            if len(chosen)==2:break
        if len(chosen)!=2:raise ValueError('Two independent training-source donors required')
        result.append(chosen)
    return result
