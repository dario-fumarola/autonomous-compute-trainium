"""CPU-only candidate:one owner per batched Muon group,replicated state.

The mathematical update and state schema remain the qualified FP32 recipe.
All participants must supply identical reduced gradients and prior state.
Native collectives,numerics,recovery and throughput are not qualified here.
"""
import torch
import torch.distributed as dist

from .parcae_runtime_scalars import RuntimeScalarMuon,ParcaeRuntimeScalarOptimizer


class OwnerBroadcastMuon(RuntimeScalarMuon):
    @torch.no_grad()
    def step(self):
        if not dist.is_initialized():raise RuntimeError('Initialized identical-replica process group required')
        rank,world=dist.get_rank(),dist.get_world_size()
        groups=self.param_groups;first=groups[0]['params'][0]
        rows=[]
        for group in groups:
            shape=tuple(group['params'][0].shape)
            rows.append([group['momentum'],group['lr']*max(1.,shape[-2]/shape[-1])**.5,
                         group['weight_decay'],group.get('beta2',.95)])
        table=torch.tensor(rows,dtype=torch.float32,device='cpu').to(first.device)
        self.last_owned_group_indices=[]
        prepared=[]
        for index,(group,values) in enumerate(zip(groups,table.unbind(0),strict=True)):
            parameters=group['params'];first=parameters[0];shape=tuple(first.shape)
            if any(p.grad is None for p in parameters):raise ValueError('Missing globally reduced Muon gradient')
            reduction=-1 if shape[-2]>=shape[-1] else -2;state=self.state[first]
            if not state:
                state['m']=torch.zeros((len(parameters),*shape),device=first.device,dtype=torch.float32)
                vshape=(len(parameters),shape[-2],1) if reduction==-1 else (len(parameters),1,shape[-1])
                state['v']=torch.zeros(vshape,device=first.device,dtype=torch.float32)
            momentum,lr,decay,beta2=values.unbind(0)
            stride=max(1,world//len(groups))
            packed=torch.stack(parameters);owner=(index*stride)%world
            prepared.append((parameters,packed,state,owner))
            if rank==owner:
                self.tensor_step(torch.stack([p.grad for p in parameters]),packed,state['m'],state['v'],
                    momentum,lr,decay,beta2,group['ns_steps'],reduction)
                self.last_owned_group_indices.append(index)
        # Independent owners finish their assigned groups before the common
        # broadcast sequence. Placing broadcasts inside the computation loop
        # would serialize owners and forfeit the intended parallel work.
        for parameters,packed,state,owner in prepared:
            # Broadcast complete moments as well as weights. Existing checkpoint
            # schemas and per-replica equality proofs therefore remain usable.
            for value in [packed,state['m'],state['v']]:dist.broadcast(value,src=owner)
            for parameter,updated in zip(parameters,packed.unbind(0),strict=True):parameter.copy_(updated)


class OwnerBroadcastOptimizerCandidate(ParcaeRuntimeScalarOptimizer):
    physical_execution='owner_broadcast_batched_fp32_muon'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.muon.__class__=OwnerBroadcastMuon

    def step(self):
        # Fail before either optimizer family mutates state.
        if not dist.is_initialized():raise RuntimeError('Initialized identical-replica process group required')
        return super().step()
