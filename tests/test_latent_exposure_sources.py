"""Boundary fixtures for source budgeting, not research text."""
import pytest
from scripts.audit_latent_exposure_sources import endpoint


class Counts:
    windows=1000
    def target_count(self,start,count):return 500*count-7


def test_smallest_complete_global_update_reaches_budget():
    assert endpoint(Counts(),128000)==3
    assert endpoint(Counts(),127993)==2


def test_exhaustion_is_rejected():
    with pytest.raises(ValueError):endpoint(Counts(),9999999)
