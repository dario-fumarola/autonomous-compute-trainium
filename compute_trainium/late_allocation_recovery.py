"""Exact same-endpoint recovery; never extend a cooled checkpoint."""
import copy
import math

from .future_span_scientific_state import SCIENCE
from .parcae_scientific_state import schedule_identity


def rebind_same_endpoint(saved, old_signature, new_signature, old_case, new_case, *, case_id, update):
    if (old_case != new_case or saved['case_id'] != case_id
            or saved['signature'] != old_signature or saved['purpose'] != SCIENCE
            or saved['qualification_only'] is not False or saved['update'] != update
            or not 0 < update < old_case['total_updates']
            or saved['cursor'] != update * old_signature['world'] * old_signature['accumulation']
            or saved['schedule'] != schedule_identity(old_case)
            or saved['stable_trunk'] != (saved['real_targets'] <= old_case['cooldown_start_targets'])):
        raise ValueError('Exact unchanged case, endpoint, schedule and scientific state required')
    old, new = copy.deepcopy(old_signature), copy.deepcopy(new_signature)
    if old.pop('plan_sha256') == new.pop('plan_sha256') or old != new:
        raise ValueError('Only the recovery plan hash may change')
    result = dict(saved)
    result['signature'] = copy.deepcopy(new_signature)
    return result


def original_deadline(now, local_deadline, composite_deadline, cap=7200):
    if not all(math.isfinite(v) for v in [now, local_deadline, composite_deadline, cap]) or not 0 < cap <= 7200:
        raise ValueError('Finite original clocks and bounded recovery cap required')
    cutoff = min(local_deadline, composite_deadline, now + cap)
    if cutoff <= now:
        raise TimeoutError('Original deadline has elapsed')
    return cutoff
