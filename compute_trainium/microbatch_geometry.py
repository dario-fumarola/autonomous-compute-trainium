"""Stack existing source windows without changing their objective or noise."""
import torch


def accumulation_for(local_batch,world=32,global_windows=512):
    if (type(local_batch) is not int or local_batch not in [1,2,4]
            or type(world) is not int or world<1 or global_windows%(world*local_batch)):
        raise ValueError('Explicit divisible fixed global batch and supported local batch required')
    return global_windows//(world*local_batch)


def stack_windows(windows):
    if not windows or any(len(w)!=7 for w in windows):
        raise ValueError('Nonempty complete source-window records required')
    for index in range(7):
        base=windows[0][index]
        if base.shape[0]!=1 or any(x[index].dtype!=base.dtype or x[index].shape!=base.shape for x in windows):
            raise ValueError('Original single-window tensor shapes/dtypes must agree')
    return tuple(torch.cat([w[i] for w in windows],dim=0) for i in range(7))


def physical_slot(logical_slot,local_batch):
    accumulation_for(local_batch)
    if type(logical_slot) is not int or not 0<=logical_slot<16:
        raise ValueError('Logical per-rank window slot must lie in0..15')
    return divmod(logical_slot,local_batch)
