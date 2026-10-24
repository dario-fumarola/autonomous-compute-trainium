"""Replayable qualification data/state and update primitives, not a research fit."""
import math
import json
from pathlib import Path

import numpy as np
import torch

from .parcae_candidate import matmul_ledger
from .parcae_update_qualification import relative_error
from .training import PackedCorpus, file_sha256


def poisson_tail(mean, cap, microstep, rank):
    """Release per-sequence sampler at local batch1, without curricula.

    Uses its rank-specific CPU seed. Prepare/save tables on one CPU runtime,
    rather than relying on identical Poisson RNG implementations across Torch
    versions. No clipping of the sampled forward count beyond minimum one.
    """
    if any(type(v) is not int or v < 1 for v in (mean, cap)) or min(microstep, rank) < 0:
        raise ValueError('Positive mean/cap and nonnegative microstep/rank required')
    seed = ((514229+microstep)*(rank+1)) % (2**31-1)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    total = max(1, int(torch.poisson(torch.tensor([float(mean)]), generator=generator)[0]))
    return total, min(total, cap)


def initial_state(config, window, seed):
    if min(window, seed) < 0:
        raise ValueError('Nonnegative window/state seed required')
    generator = torch.Generator(device='cpu').manual_seed((seed+window) % (2**63-1))
    state = torch.empty((1, 1024, config.dim), dtype=torch.float32)
    std = math.sqrt(2/(5*config.dim))
    return torch.nn.init.trunc_normal_(state, std=std, a=-3*std, b=3*std, generator=generator)


def configure_optimizer(optimizer, update):
    """Conservative qualification schedule, not the publication LR recipe."""
    rate = 1e-4*min((update+1)/20, 1.)
    for group in optimizer.param_groups:
        group['lr'] = rate*group.get('base_lr', 1.)
        if group['kind'] == 'muon':
            group['momentum'] = .85+.1*min((update+1)/20, 1.)
    return rate


def finish_gradients(model, *, scale, clip=1.):
    if scale not in (1, 8192) or clip <= 0:
        raise ValueError('Declared scale and positive clipping required')
    for parameter in model.parameters():
        if parameter.grad is None or parameter.grad.dtype != torch.float32:
            raise ValueError('Every trainable FP32 master must receive a gradient')
        if scale != 1:
            parameter.grad.div_(scale)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, foreach=False)
    value = float(norm.detach().cpu())
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite full global gradient; do not step')
    return value


def model_work(config, depths, context=1024):
    """Exact host depth table work; one sequence/window per table entry."""
    return sum(matmul_ledger(config, context, 1, recurrent=True,
        total_steps=int(n), backward_cap=int(k))['training_matmul_flops'] for n,k in depths)


def validate_resume(saved, signature):
    if saved['qualification_only'] is not True or saved['signature'] != signature:
        raise ValueError('Qualification signature changed; no continuation')
    count = signature['world']*signature['accumulation']
    if saved['cursor'] != saved['update']*count:
        raise ValueError('Checkpoint source cursor and update disagree')
    if min(saved['real_targets'], saved['model_matmuls'], saved['optimizer_matmuls']) < 0:
        raise ValueError('Invalid exposure/work counters')


def map_after_shared_validation(directory, manifest_sha256):
    """Only after rank0 validates all files and a successful DDP barrier.

    Every rank still checks the identical frozen manifest and dimensions. This
    avoids a large full-file hash pass on every replica, never changes packing, and
    does not replace the required full validation by rank0 in each process.
    """
    directory = Path(directory)
    if file_sha256(directory/'manifest.json') != manifest_sha256:
        raise ValueError('Shared validated packing manifest changed')
    metadata = json.loads((directory/'manifest.json').read_text())
    result = PackedCorpus.__new__(PackedCorpus)
    for key in ('context','seed','windows','real_targets','domain_counts','fingerprint','order_fingerprint'):
        setattr(result, key, metadata[key])
    for name in ('tokens','domains','prefix_counts'):
        setattr(result, name, torch.from_numpy(np.load(directory/(name+'.npy'), mmap_mode='c')))
    if (result.tokens.shape != (result.windows*result.context+1,) or
            result.domains.shape != result.tokens.shape or
            result.prefix_counts.shape != (result.windows+1,3)):
        raise ValueError('Mapped packing dimensions differ')
    return result


def gradient_comparison(expected, actual):
    def family(name):
        if name=='transformer.wte.weight':return 'tied_input_output'
        if name.startswith('value_embeds.'):return 'value_embeddings'
        if name.startswith('transformer.adapter.') or name.startswith('transformer.C.'):
            return 'recurrent_injection_readout'
        return 'transformer_body'
    overall=relative_error(expected,actual)
    groups={}
    for category in sorted({family(n) for n in expected}):
        names=[n for n in expected if family(n)==category]
        groups[category]=relative_error({n:expected[n] for n in names},{n:actual[n] for n in names})
    return {'overall':overall,'families':groups,
            'passes_two_percent_gate':overall['relative_l2']<=.02 and all(v['relative_l2']<=.02 for v in groups.values())}
