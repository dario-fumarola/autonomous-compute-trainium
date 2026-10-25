"""Prospective work-matched continuation geometry; no training admission."""
from .parcae_scientific_schedule import matched_endpoint, source_endpoint, cooldown_start


def control_branch(work, prefix_counts, budget, *, origin_update,
                   global_windows=512, warmup_targets=50_000_000):
    """Pay the complete inherited prefix and finish an independent cooldown.

    Use the largest complete update count that does not exceed the candidate's
    work. The next ledger row must remain provisioned for the scientific worker.
    No checkpoint is adopted and no already annealed trajectory is accepted.
    """
    if type(origin_update) is not int or origin_update < 1:
        raise ValueError('Positive stable branch origin required')
    endpoint = matched_endpoint(work, budget)
    end = endpoint['updates']
    if end >= len(work):
        raise ValueError('Provision a following work row for the scientific worker')
    decay = cooldown_start(end)
    if origin_update > decay:
        raise ValueError('Branch origin is later than the new cooldown start')
    origin = source_endpoint(prefix_counts, origin_update, global_windows)
    cooldown = source_endpoint(prefix_counts, decay, global_windows)
    final = source_endpoint(prefix_counts, end, global_windows)
    if (type(warmup_targets) is not int or not 0 < warmup_targets < cooldown['real_targets']
            or not origin['real_targets'] <= cooldown['real_targets'] < final['real_targets']
            or any(a > b or b > c for a, b, c in zip(origin['domain_targets'],
                cooldown['domain_targets'], final['domain_targets'], strict=True))):
        raise ValueError('Ordered source exposure, warmup and independent cooldown required')
    inherited = sum(map(int, work[:origin_update]))
    return {'matched_work': endpoint, 'origin_update': origin_update,
            'cooldown_start_update': decay, 'total_updates': end,
            'origin_source': origin, 'cooldown_source': cooldown, 'final_source': final,
            'inherited_matrix_ops_charged': inherited,
            'additional_branch_matrix_ops': endpoint['executed_matmuls'] - inherited,
            'checkpoint_adopted': False, 'training_launched': False,
            'scope': 'Complete conceptual model cost includes the shared prefix; incremental campaign cost is separate. Requires an actually stable, fully recovered parent and a separately qualified branch registration before launch. Not chip-time matching or an efficiency result.'}
