"""Stateless example routing for a static-per-worker depth mixture.

No corpus generation: these functions assign existing packed natural windows.
The model parameters are shared across all ranks by ordinary DDP.
"""
from functools import lru_cache

_MASK = (1 << 64) - 1


def rank_depths(world, policy):
    if world < 3:
        raise ValueError('At least three ranks required')
    if policy == 'ordinary':
        return (1,) * world
    if policy == 'fixed':
        return (2,) * world
    if policy != 'mixture':
        raise ValueError('Unknown depth policy')
    outer = world // 3
    return (1,) * outer + (2,) * (world - 2 * outer) + (3,) * outer


@lru_cache(maxsize=2)
def window_permutation(cursor, count, seed):
    """Explicit SplitMix64/Fisher-Yates, independent of Python RNG versions."""
    if cursor < 0 or count <= 0 or seed < 0:
        raise ValueError('Invalid routing configuration')
    state = (seed ^ cursor) & _MASK
    order = list(range(count))
    for end in range(count - 1, 0, -1):
        state = (state + 0x9E3779B97F4A7C15) & _MASK
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
        value ^= value >> 31
        # Tiny modulo bias is disclosed; routing is deterministic, not a source RNG.
        selected = value % (end + 1)
        order[end], order[selected] = order[selected], order[end]
    return tuple(order)


def routed_indices(cursor, micro, rank, world, batch, accumulation, seed=271828):
    if not (0 <= rank < world and 0 <= micro < accumulation and batch > 0):
        raise ValueError('Invalid worker/microbatch')
    order = window_permutation(cursor, world * batch * accumulation, seed)
    first = (micro * world + rank) * batch
    return [cursor + order[i] for i in range(first, first + batch)]


def mixture_step_flops(config, world, batch, accumulation, context):
    """Every actual block application and every vocabulary projection is counted."""
    from dataclasses import replace
    return sum(replace(config, repetitions=d).flops(context, context, batch * accumulation)
               ['training_matmul_flops'] for d in rank_depths(world, 'mixture'))
