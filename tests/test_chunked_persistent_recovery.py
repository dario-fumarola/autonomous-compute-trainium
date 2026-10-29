import pytest
from scripts.run_chunked_persistent_recovery import retained_deadline_plan


def test_recovery_preserves_original_budget_and_scientific_settings():
    old=dict(identity='old',absolute_deadline_unix=2000.,phase_seconds_limit=1200,
        cpu_phase_seconds_limit=900,seed=40481,depth_schedule=[0,1,2,3],
        per_graph_repeated_update_seconds_limit=1.,generated_to_full_ordinary_seconds_ratio_limit=3.,
        scientific_fit_admitted=False,retained_for_science=False)
    p=retained_deadline_plan(old,1000.)
    assert p['absolute_deadline_unix']==2000. and p['seconds_limit']==1000
    for key in old:
        if key!='identity':assert p[key]==old[key]
    p['depth_schedule'][0]=99
    assert old['depth_schedule'][0]==0


def test_recovery_cannot_renew_expired_or_too_short_budget():
    for now in (1401.,2001.):
        with pytest.raises(TimeoutError):retained_deadline_plan(dict(absolute_deadline_unix=2000.),now)
