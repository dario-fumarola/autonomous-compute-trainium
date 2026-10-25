"""Execution identities and metadata-only adoption of unchanged training state."""
from .parcae_scientific_state import signature


def gradient_identity(arm, readout):
    if arm not in ['ordinary', 'fixed1', 'poisson2'] or readout not in ['stable1B', 'matched', 'equal']:
        raise ValueError('Only declared scientific gradient stages are recoverable')
    return f'low-depth-main-r2-gradient-{arm}-{readout.lower()}'


def rebind_checkpoint(saved, old_sha, old_plan, new_sha, new_plan, case_id):
    """Return a shallow metadata copy; never mutate a tensor or scientific field."""
    for key in ['purpose', 'model', 'precision', 'accumulation', 'routing_seed',
                'data', 'initial_states', 'cases']:
        if old_plan[key] != new_plan[key]:
            raise ValueError('Recovery changed scientific configuration: '+key)
    old = signature(old_sha, old_plan, old_plan['cases'][case_id])
    new = signature(new_sha, new_plan, new_plan['cases'][case_id])
    if saved['case_id'] != case_id or saved['signature'] != old:
        raise ValueError('Exact predecessor checkpoint identity required')
    if new != {**old, 'plan_sha256': new_sha}:
        raise ValueError('Recovery may rebind only the execution-plan signature')
    return {**saved, 'signature': new}
