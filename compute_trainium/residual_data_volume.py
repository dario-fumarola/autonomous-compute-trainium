"""Exact branch continuation and frozen fresh/repeated source schedules."""
import torch


def source_indices(update,batch,unique_views):
    if update<0 or batch<1 or unique_views<batch or unique_views%batch:raise ValueError('Registered complete source batches required')
    return torch.arange(update*batch,(update+1)*batch)%unique_views


def restore_branch(path,model,optimizer,generator):
    saved=torch.load(path,weights_only=True);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);generator.set_state(saved['rng'])
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in saved['model'].items()):raise AssertionError('Branch restore changed parameters')
    for index,fields in saved['optimizer']['state'].items():
        for key,value in fields.items():
            other=optimizer.state_dict()['state'][index][key]
            if isinstance(value,torch.Tensor) and not torch.equal(value,other):raise AssertionError('Branch optimizer restore changed moments/counter')
    if not torch.equal(saved['rng'],generator.get_state()):raise AssertionError('Sampling RNG restore changed')
    return saved['updates']
