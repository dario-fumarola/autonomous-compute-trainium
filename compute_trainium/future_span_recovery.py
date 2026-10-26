"""Rebind only a stable checkpoint's plan identity for explicit recovery."""
import copy

from .future_span_scientific_state import SCIENCE
from .parcae_scientific_state import schedule_identity


def rebind_stable_checkpoint(saved, old_signature, new_signature, old_case, *, update):
    """No model/optimizer/RNG/counter change; caller separately validates bytes."""
    if (saved['signature'] != old_signature or saved['purpose'] != SCIENCE
            or saved['qualification_only'] is not False or saved['stable_trunk'] is not True
            or saved['update'] != update or saved['cursor'] != update*512
            or saved['schedule'] != schedule_identity(old_case)
            or saved['real_targets'] > old_case['cooldown_start_targets']):
        raise ValueError('Exact scientific stable prefix required for recovery')
    old = copy.deepcopy(old_signature); new = copy.deepcopy(new_signature)
    if old.pop('plan_sha256') == new.pop('plan_sha256') or old != new:
        raise ValueError('Recovery can change only the registered plan hash')
    result = dict(saved)
    result['signature'] = copy.deepcopy(new_signature)
    return result
