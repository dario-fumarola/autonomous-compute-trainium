"""Complete local-shard actor/scorer/AdamW/RNG state for workspace execution."""
import io
import zipfile
import torch
from .scale_sharded_candidate import local_parameter
from .workspace_streaming_checkpoint import save_streaming,restore_streaming


def complete_state_tensors(model,scorer,optimizer):
    for label,current in [('model',model),('scorer',scorer)]:
        if current is None:continue
        for name,p in current.named_parameters():yield label+'.'+name,local_parameter(p)
    for index,alias in enumerate(optimizer.local):
        for name,value in sorted(optimizer.adam.state[alias].items()):
            if not isinstance(value,torch.Tensor):raise TypeError('Uncovered non-tensor optimizer state')
            yield f'adam.{index}.{name}',value


def restore_complete_state(path,model,scorer,optimizer,generator):
    with zipfile.ZipFile(path) as archive:receipt=torch.load(io.BytesIO(archive.read('metadata.pt')),weights_only=True)
    metadata=receipt['metadata']
    if metadata.get('scorer_present',True)!=(scorer is not None):raise ValueError('Exact optional frozen-scorer coverage required')
    if metadata['parameter_names']!=[n for n,p in optimizer.named]:raise ValueError('Complete exact parameter order required')
    if len(optimizer.adam.param_groups)!=len(metadata['param_groups']):raise ValueError('Complete optimizer groups required')
    for group,values in zip(optimizer.adam.param_groups,metadata['param_groups'],strict=True):group.update(values)
    fused=optimizer.adam.param_groups[0].get('fused',False)
    for index,alias in enumerate(optimizer.local):
        state=optimizer.adam.state[alias]
        for field in ['exp_avg','exp_avg_sq','step']:
            sig=receipt['signatures'][f'adam.{index}.{field}']
            dtype=getattr(torch,sig['dtype'].split('.')[-1])
            state[field]=torch.empty(tuple(sig['shape']),device=(alias.device if field!='step' or fused else 'cpu'),dtype=dtype)
    restored=restore_streaming(path,complete_state_tensors(model,scorer,optimizer))
    optimizer.update=restored['update'];torch.set_rng_state(restored['torch_rng']);generator.set_state(restored['sampling_rng'])
    if not torch.equal(torch.get_rng_state(),metadata['torch_rng']) or not torch.equal(generator.get_state(),metadata['sampling_rng']):raise AssertionError('Complete restored RNG differs')
    return restored


def save_complete_state(path,model,scorer,optimizer,generator,*,extra,compress=False):
    metadata=dict(extra,update=optimizer.update,parameter_names=[n for n,p in optimizer.named],
        scorer_present=scorer is not None,
        param_groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.adam.param_groups],
        torch_rng=torch.get_rng_state(),sampling_rng=generator.get_state())
    return save_streaming(path,complete_state_tensors(model,scorer,optimizer),metadata,compress=compress)


def release_compiled_calls(*models):
    """Checkpoint needs tensor shards,not compiled forward/backward wrappers."""
    for model in models:
        if model is None:continue
        for module in model.modules():module._compiled_call_impl=None
    from .workspace_native_objective import drain_runtime
    first=next(model for model in models if model is not None)
    drain_runtime(next(first.parameters()).device)


def save_state_waves(path,model,scorer,optimizer,generator,*,extra,wave_size=8,save_ranks=None):
    """Bound transient CPU transfer/compression across ranks;no new updates."""
    import torch.distributed as dist
    rank,world=dist.get_rank(),dist.get_world_size()
    count=world if save_ranks is None else save_ranks
    if type(wave_size) is not int or wave_size<1 or not 0<count<=world:raise ValueError('Bounded nonempty checkpoint waves required')
    result=None
    for start in range(0,count,wave_size):
        if start<=rank<min(start+wave_size,count):
            result=save_complete_state(path,model,scorer,optimizer,generator,extra=extra,compress=True)
        dist.barrier()
        from .workspace_native_objective import drain_runtime
        drain_runtime(next(model.parameters()).device)
    return result


@torch.no_grad()
def restore_model_shards(path,model,*,label):
    from .workspace_streaming_checkpoint import tensor_hash
    with zipfile.ZipFile(path) as archive:
        receipt=torch.load(io.BytesIO(archive.read('metadata.pt')),weights_only=True)
        if receipt['metadata']['parameter_names']!=[n for n,p in model.named_parameters()]:raise ValueError('Exact model shard coverage/order required')
        for name,p in model.named_parameters():
            sig=receipt['signatures'][label+'.'+name]
            value=torch.load(io.BytesIO(archive.read(sig['member'])),weights_only=True)
            if tensor_hash(value)!=sig['sha256']:raise AssertionError('Original model shard changed')
            target=local_parameter(p)
            if target.shape!=value.shape or target.dtype!=value.dtype:raise ValueError('Exact model shard shape/dtype required')
            target.copy_(value.to(target.device))
