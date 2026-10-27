"""Complete local-shard actor/scorer/AdamW/RNG state for workspace execution."""
import io
import zipfile
import torch
from .scale_sharded_candidate import local_parameter
from .workspace_streaming_checkpoint import save_streaming,restore_streaming


def complete_state_tensors(model,scorer,optimizer):
    for label,current in [('model',model),('scorer',scorer)]:
        for name,p in current.named_parameters():yield label+'.'+name,local_parameter(p)
    for index,alias in enumerate(optimizer.local):
        for name,value in sorted(optimizer.adam.state[alias].items()):
            if not isinstance(value,torch.Tensor):raise TypeError('Uncovered non-tensor optimizer state')
            yield f'adam.{index}.{name}',value


def restore_complete_state(path,model,scorer,optimizer,generator):
    with zipfile.ZipFile(path) as archive:receipt=torch.load(io.BytesIO(archive.read('metadata.pt')),weights_only=True)
    metadata=receipt['metadata']
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


def save_complete_state(path,model,scorer,optimizer,generator,*,extra):
    metadata=dict(extra,update=optimizer.update,parameter_names=[n for n,p in optimizer.named],
        param_groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.adam.param_groups],
        torch_rng=torch.get_rng_state(),sampling_rng=generator.get_state())
    return save_streaming(path,complete_state_tensors(model,scorer,optimizer),metadata)
