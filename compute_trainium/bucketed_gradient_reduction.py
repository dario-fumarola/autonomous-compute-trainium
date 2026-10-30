"""Prospective reduction/clip batching; not enabled in any scientific trainer.

The callback must synchronously sum a dense buffer across the declared ranks.
Gradient averaging and global norm clipping retain their existing semantics.
The existing per-parameter norm decomposition is retained; Native qualification
is still mandatory for communication, views and floating-point behavior.
"""
import math
import torch


def reduce_and_clip(parameters, reduce_sum, world, *, max_norm=1., bucket_bytes=32*1024**2):
    if type(world) is not int or world < 1:
        raise ValueError('Positive world size required')
    if type(bucket_bytes) is not int or bucket_bytes < 1:
        raise ValueError('Positive bucket allowance required')
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError('Positive finite clipping threshold required')
    parameters = list(parameters)
    if not parameters or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError('Nonempty unique parameter list required')
    gradients = []
    for p in parameters:
        g = p.grad
        if g is None or g.is_sparse or g.dtype not in (torch.float32,torch.float64):
            raise ValueError('Complete dense FP32/FP64 gradients required')
        if gradients and (g.dtype != gradients[0].dtype or g.device != gradients[0].device):
            raise ValueError('Uniform gradient device and dtype required')
        gradients.append(g.detach())
    groups = []; pending = []; size = 0
    for p,g in zip(parameters,gradients):
        n = g.numel()*g.element_size()
        if pending and size+n > bucket_bytes:
            groups.append(pending); pending = []; size = 0
        pending.append((p,g)); size += n
        if size >= bucket_bytes:
            groups.append(pending); pending = []; size = 0
    if pending: groups.append(pending)
    buffers = []; views = []
    with torch.no_grad():
        for group in groups:
            flat = (group[0][1].reshape(-1) if len(group)==1 else
                    torch.cat([g.reshape(-1) for _,g in group]))
            reduce_sum(flat)
            flat.div_(world)
            buffers.append(flat)
            offset = 0
            for p,g in group:
                p.grad = flat[offset:offset+g.numel()].view_as(p)
                views.append(p.grad); offset += g.numel()
        # Large concatenated FP32 norms have different accumulation error.
        # Preserve the trainer's original per-parameter decomposition.
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g,2) for g in views]),2)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Non-finite global gradient norm')
        coefficient = (max_norm/(norm+1e-6)).clamp(max=1.)
        for flat in buffers: flat.mul_(coefficient)
    return norm, len(buffers)
