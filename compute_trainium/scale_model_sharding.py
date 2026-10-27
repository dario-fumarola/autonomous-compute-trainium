"""Direct CPU-to-local loading and bounded complete-matrix owner workspaces."""
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, Replicate

from .scale_sharded_candidate import shard_model, ShardedExactOptimizer, local_parameter
from .parcae_optimizer_precise import muon_tensor_step_fp32


def shard_model_from_cpu(model, mesh, state, device):
    """Construct local storage from a meta model; never allocate global device C0."""
    aliases = {}; world=dist.get_world_size();rank=dist.get_rank()
    for module_name, module in model.named_modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is None:continue
            key=id(parameter)
            if key not in aliases:
                full_name=module_name+'.'+name if module_name else name
                value=state[full_name]
                if tuple(value.shape)!=tuple(parameter.shape) or value.dtype!=torch.float32 or value.device.type!='cpu':
                    raise ValueError('Exact CPU FP32 initializer shape required')
                placement=Shard(0) if parameter.shape[0]%world==0 else Replicate()
                local=value.chunk(world,0)[rank] if placement.is_shard() else value
                aliases[key]=nn.Parameter(DTensor.from_local(local.contiguous().to(device,copy=True),mesh,[placement],
                    shape=parameter.shape,stride=parameter.stride(),run_check=False))
            module._parameters[name]=aliases[key]
        for name, buffer in list(module._buffers.items()):
            if buffer is not None:
                full_name=module_name+'.'+name if module_name else name
                module._buffers[name]=state[full_name].to(device)
    # Reuse the exact property installation without redistributing the tensors.
    from .scale_sharded_candidate import _WRAPPERS,_gather_parameter
    import sys
    for module in model.modules():
        names=[name for name,p in module._parameters.items() if p is not None]
        if not names:continue
        key=(module.__class__,tuple(names))
        if key not in _WRAPPERS:
            label='ScaleShard'+module.__class__.__name__+str(len(_WRAPPERS))
            props={name:property(lambda self,n=name:_gather_parameter(self,n)) for name in names}
            _WRAPPERS[key]=type(label,(module.__class__,),props)
            # Same globally importable location as the original wrapper cache.
            from . import scale_sharded_candidate
            setattr(scale_sharded_candidate,label,_WRAPPERS[key])
        module.__class__=_WRAPPERS[key]
    return model


class StreamingOwnerOptimizer(ShardedExactOptimizer):
    """Same update; nonowners retain one full matrix, not a packed full group."""
    @torch.no_grad()
    def step(self):
        if any(p.grad is None for _,p in self.named):raise ValueError('All gradients required before updating')
        for alias,parameter in zip(self.local_adam,self.groups[0]['params'],strict=True):
            alias.grad=parameter.grad.to_local()
        self.adamw.param_groups[0].update({k:v for k,v in self.groups[0].items() if k!='params'})
        self.adamw.step()
        for index,group in enumerate(self.groups[1:]):
            owner=index%self.world;parameters=group['params'];shape=tuple(parameters[0].shape)
            weights=[];gradients=[]
            for parameter in parameters:
                weight=parameter.full_tensor();gradient=parameter.grad.full_tensor()
                if self.rank==owner:weights.append(weight);gradients.append(gradient)
                del weight,gradient
            if self.rank==owner:
                packed=torch.stack(weights);gradient=torch.stack(gradients)
                del weights,gradients
                reduction=-1 if shape[-2]>=shape[-1] else -2
                if index not in self.muon_state:
                    vshape=(len(parameters),shape[-2],1) if reduction==-1 else (len(parameters),1,shape[-1])
                    self.muon_state[index]=dict(m=torch.zeros_like(packed),v=torch.zeros(vshape,device=packed.device,dtype=torch.float32))
                state=self.muon_state[index]
                scalars=torch.tensor([group['momentum'],group['lr']*max(1.,shape[-2]/shape[-1])**.5,
                    group['weight_decay'],group.get('beta2',.95)],dtype=torch.float32).to(packed.device)
                muon_tensor_step_fp32(gradient,packed,state['m'],state['v'],*scalars.unbind(0),group['ns_steps'],reduction)
                del gradient
            for offset,parameter in enumerate(parameters):
                value=packed[offset].contiguous() if self.rank==owner else torch.empty(shape,device=local_parameter(parameter).device)
                dist.broadcast(value,src=owner)
                local=value.chunk(self.world,0)[self.rank] if parameter.placements[0].is_shard() else value
                local_parameter(parameter).copy_(local)
                del local,value
            if self.rank==owner:del packed
        self.update+=1
