"""Explicit cross-plan identity transport for unchanged stable scientific state."""
from .future_span_scientific_state import SCIENCE, signature, validate_checkpoint
from scripts.qualify_native import digest


def transport_stable(saved, *, source_plan, source_plan_sha, source_case_id,
                     source_checkpoint_sha, destination_plan, destination_plan_sha,
                     destination_case_id, expected_counters):
    """Change only the plan binding and append provenance; retain past history.

    Callers must verify both immutable plans and the actual source file hash.
    This operation is not a training admission or numerical restart test.
    """
    if source_plan_sha == destination_plan_sha:
        raise ValueError('Same-plan continuation needs no identity transport')
    if source_plan['purpose'] != SCIENCE or destination_plan['purpose'] != SCIENCE:
        raise ValueError('Only scientific-to-scientific stable state transport is allowed')
    source = source_plan['cases'][source_case_id]
    destination = destination_plan['cases'][destination_case_id]
    original_signature = signature(source_plan_sha, source_plan, source)
    new_signature = signature(destination_plan_sha, destination_plan, destination)
    if saved['case_id'] != source_case_id:
        raise ValueError('Exact original scientific case required')
    left = {k: v for k, v in original_signature.items() if k != 'plan_sha256'}
    right = {k: v for k, v in new_signature.items() if k != 'plan_sha256'}
    if left != right:
        raise ValueError('Transport cannot change model, objective, data, initializer, RNG, precision or execution')
    validate_checkpoint(saved, source_case_id, source, original_signature, **expected_counters)
    if not saved['stable_trunk']:
        raise ValueError('An annealed checkpoint cannot become a new stable prefix')
    if destination.get('parent') != {'case_id': source_case_id, 'update': saved['update']}:
        raise ValueError('Exact stable parent must be registered in the destination')
    if (not 0 < destination['warmup_targets'] < destination['cooldown_start_targets'] < destination['end_targets']
            or not saved['update'] <= destination['cooldown_start_update'] < destination['total_updates']):
        raise ValueError('Continuation must have its own ordered future cooldown')
    state_digest = digest({'model': dict(sorted(saved['model'].items())), 'optimizer': saved['optimizer']})
    if state_digest != saved['model_optimizer_digest']:
        raise ValueError('Original complete model and optimizer state differs')
    if len(saved['cpu_rng_by_rank']) != 32:
        raise ValueError('All original per-rank RNG states are required')
    rng_digest = digest(saved['cpu_rng_by_rank'])
    provenance = {'operation': 'plan_identity_only',
        'source_plan_sha256': source_plan_sha, 'source_case_id': source_case_id,
        'source_checkpoint_sha256': source_checkpoint_sha,
        'destination_plan_sha256': destination_plan_sha,
        'complete_model_optimizer_digest': state_digest, 'cpu_rng_digest': rng_digest,
        'inherited_update': saved['update'], 'inherited_cursor': saved['cursor'],
        'inherited_real_targets': saved['real_targets'],
        'inherited_matrix_ops': saved['estimated_dense_matrix_ops']}
    result = dict(saved)
    result['signature'] = new_signature
    result['continuation_provenance'] = [*saved.get('continuation_provenance', []), provenance]
    validate_checkpoint(result, destination_case_id, destination, new_signature, **expected_counters)
    # No tensor tree or historical schedule/counter is copied or mutated here.
    assert result['model'] is saved['model'] and result['optimizer'] is saved['optimizer']
    assert result['cpu_rng_by_rank'] is saved['cpu_rng_by_rank']
    return result, provenance
