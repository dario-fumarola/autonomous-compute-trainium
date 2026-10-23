"""Future execution resize that preserves each window's recurrent depth.

This is not used by the frozen campaign and does not migrate tensor state.
Native numerical/recovery qualification is required before using a resize.
"""
from .checkpoint_migration import migration_signature
from .depth_mixture import rank_depths


def resize_depth_signature(old, *, world, batch):
    if old['routing'] != 'splitmix64_fisher_yates_v1':
        raise ValueError('Unsupported source routing')
    policy = old['depth_policy']
    before = rank_depths(old['world'], policy)
    after = rank_depths(world, policy)
    if list(before) != old['rank_depths']:
        raise ValueError('Recorded rank depths do not match the old policy')
    # Routing permutes the global accumulated source windows, then distributes
    # contiguous batches within each microbatch. Thus equality of this vector
    # and microbatch size preserves the depth of every original source window.
    left = tuple(d for d in before for _ in range(old['batch']))
    right = tuple(d for d in after for _ in range(batch))
    if left != right:
        raise ValueError('Resize changes source-window depth assignments')
    new = migration_signature(old, world=world, batch=batch,
        accumulation=old['accumulation'], corpus_sha=old['corpus_sha256'],
        packing_sha=old['packing_sha256'], optimizer=old['optimizer'])
    new['rank_depths'] = list(after)
    return new
