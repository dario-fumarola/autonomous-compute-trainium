"""Supported exits distinguish earlier latent writes from query refinement."""
import torch
from torch.nn import functional as F
from .residual_workspace import ResidualWorkspace,unit


def refine_current(cell,state,memory,anchors,*,reference=False):
    q=cell.q(unit(state));k=cell.k(unit(memory));v=cell.v(unit(memory));current_k=cell.k(unit(state));current_v=cell.v(unit(state));positions=torch.arange(memory.shape[1],device=memory.device);mask=positions[None,:]<=anchors[:,None];scale=state.shape[-1]**-.5
    if reference:
        same=positions[None,:]==anchors[:,None];keys=torch.where(same[None,:,:,None],current_k[:,:,None,:],k[:,None,:,:]);values=torch.where(same[None,:,:,None],current_v[:,:,None,:],v[:,None,:,:]);scores=(q[:,:,None,:]*keys).sum(-1)*scale;weights=scores.masked_fill(~mask[None,:,:],float('-inf')).softmax(-1);read=(weights[:,:,:,None]*values).sum(-2)
    else:
        index=anchors[None,:,None].expand(state.shape[0],-1,1);scores=(q@k.transpose(-1,-2))*scale;scores=scores.scatter(-1,index,(q*current_k).sum(-1,keepdim=True)*scale);weights=scores.masked_fill(~mask,float('-inf')).softmax(-1);read=weights@v+weights.gather(-1,index)*(current_v-v.index_select(1,anchors))
    state=state+cell.o(read)
    return state+cell.down(F.silu(cell.up(unit(state))))


class WrittenMemoryWorkspace(ResidualWorkspace):
    def __init__(self,hidden_size=2048,width=128,*,writes=False):
        super().__init__(hidden_size,width,mode='one');self.writes=writes

    def forward(self,features,anchors,*,depth,disabled=False,suppress_earlier_writes=False,reference=False):
        if depth not in (1,2):raise ValueError('Only explicitly trained exits are supported')
        base=features.index_select(1,anchors)
        if disabled:return base
        memory=self.input(unit(features));state=memory.index_select(1,anchors)
        if depth==1 or not self.writes:
            state=refine_current(self.cell,state,memory,anchors,reference=reference)
        else:
            updated=refine_current(self.cell,memory,memory,torch.arange(memory.shape[1],device=features.device),reference=reference);state=updated.index_select(1,anchors)
        if depth==2:
            # Both conditions preserve the UPDATED current-token slot. Only
            # earlier states differ. The intervention still computes the writes.
            supplied=updated if self.writes and not suppress_earlier_writes else memory
            state=refine_current(self.cell,state,supplied,anchors,reference=reference)
        return base+self.output(unit(state))


def balanced_depth_schedule(updates=2048,*,seed=7299,block=256):
    import random
    if block%2 or updates%block:raise ValueError('Complete balanced depth blocks required')
    rng=random.Random(seed);result=[]
    for _ in range(updates//block):
        depths=[1]*(block//2)+[2]*(block//2);rng.shuffle(depths);result.extend(depths)
    return result
