"""Bound transient host copies during full model/optimizer/RNG capture."""
import ctypes
import gc
import json
from pathlib import Path
import torch
import torch.distributed as dist
from .scale_sharded_fixture import cpu_tree


def capture_groups(world,rank_batch):
    if type(world) is not int or world<1 or type(rank_batch) is not int or not 1<=rank_batch<=4:
        raise ValueError('Positive world and at most four simultaneous host copies')
    return [range(start,min(world,start+rank_batch)) for start in range(0,world,rank_batch)]


def capture_replicas(model,optimizer,*,rank,world,rank_batch,output,update,fallback_ops,on_primary):
    from scripts.qualify_native import digest
    if not 0<=rank<world:raise ValueError('Invalid rank')
    groups=capture_groups(world,rank_batch);output=Path(output);state_hash=None
    # Gradients are not checkpoint state. Release them before allocating host
    # copies; this does not change model,AdamW moments,hyperparameters or RNG.
    optimizer.zero_grad(set_to_none=True)
    dist.barrier()
    for members in groups:
        if rank in members:
            state=cpu_tree(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()))
            state_hash=digest(state)
            (output/f'replica-{rank}.json').write_text(json.dumps(dict(update=update,digest=state_hash,fallback_ops=fallback_ops)))
            if rank==0:on_primary(state,state_hash)
            del state
            gc.collect()
            libc=ctypes.CDLL(None)
            if hasattr(libc,'malloc_trim'):libc.malloc_trim(0)
        dist.barrier()
    if rank==0:
        rows=[json.loads((output/f'replica-{r}.json').read_text()) for r in range(world)]
        if len({r['digest'] for r in rows})!=1 or any(r['update']!=update for r in rows):raise AssertionError('Replica full states differ')
        if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in rows):raise AssertionError('Unexpected fallback')
    dist.barrier()
    return state_hash
