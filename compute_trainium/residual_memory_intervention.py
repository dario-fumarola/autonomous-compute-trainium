"""Zero-update memory-content interventions with an unchanged original query."""
import torch
from torch.nn import functional as F
from .residual_workspace import unit


def read_with_memory(model,features,anchors,*,donor=None,self_only=False):
    if model.mode!='one':raise ValueError('Only the trained one-step exit is admitted')
    if donor is not None and donor.shape!=features.shape:raise ValueError('Matched memory geometry required')
    if donor is not None and self_only:raise ValueError('Separate registered interventions')
    base=features.index_select(1,anchors);original=model.input(unit(features));state=original.index_select(1,anchors)
    positions=torch.arange(features.shape[1],device=features.device);current=positions[None,:]==anchors[:,None]
    supplied=original if donor is None else model.input(unit(donor))
    # Each query keeps its own original slot,even when that slot is an earlier
    # memory position for a different query. No original earlier K/V survives.
    memory=torch.where(current[None,:,:,None],original[:,None,:,:],supplied[:,None,:,:])
    cell=model.cell;q=cell.q(unit(state));k=cell.k(unit(memory));v=cell.v(unit(memory))
    allowed=current if self_only else positions[None,:]<=anchors[:,None]
    scores=(q[:,:,None,:]*k).sum(-1)*(state.shape[-1]**-.5)
    weights=scores.masked_fill(~allowed[None,:,:],float('-inf')).softmax(-1)
    state=state+cell.o((weights[:,:,:,None]*v).sum(-2))
    state=state+cell.down(F.silu(cell.up(unit(state))))
    return base+model.output(unit(state))


def choose_donors(records,*,seed=7291):
    """Two distinct same-kind/domain source groups;outcomes never enter selection."""
    import hashlib
    selected={};excluded=[]
    for i,row in enumerate(records):
        candidates=[j for j,v in enumerate(records) if v['stratum']==row['stratum'] and v['group']!=row['group'] and v['length']>=row['length']]
        candidates.sort(key=lambda j:hashlib.sha256(f'{seed}:{i}:{j}'.encode()).hexdigest())
        chosen=[];groups=set()
        for j in candidates:
            if records[j]['group'] in groups:continue
            chosen.append(j);groups.add(records[j]['group'])
            if len(chosen)==2:break
        if len(chosen)==2:selected[str(i)]=chosen
        else:excluded.append(i)
    return selected,excluded
