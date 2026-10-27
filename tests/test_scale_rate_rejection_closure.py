from copy import deepcopy
from scripts.close_scale_original_rate_grid import valid_rejected_observation


def test_rejection_closure_requires_complete_actual_numeric_failure():
    base=dict(windows=512,gradient={'passes_two_percent_gate':False},native_minus_cpu_nll=1e-6,
              signature_matches=True,source_cursor_matches=True,real_targets_match=True,
              native_clipped=True,model_unchanged=True)
    assert valid_rejected_observation(base)
    for key,bad in [('windows',511),('signature_matches',False),('model_unchanged',False),
                    ('native_minus_cpu_nll',float('nan'))]:
        changed=deepcopy(base);changed[key]=bad;assert not valid_rejected_observation(changed)
    changed=deepcopy(base);changed['gradient']['passes_two_percent_gate']=True
    assert not valid_rejected_observation(changed)
