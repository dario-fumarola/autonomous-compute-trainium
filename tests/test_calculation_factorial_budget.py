import pytest
from scripts.run_calculation_factorial import audit_start_for


def test_both_audit_batches_are_outside_fit_and_available():
    assert audit_start_for('original',11718750)==10000000
    assert audit_start_for('enriched',951172)==950000


@pytest.mark.parametrize('condition,windows,updates',[
    ('enriched',950127,1957),('enriched',951172,7422),
    ('original',9999999,1957),('original',11718750,0),
])
def test_rejects_missing_audit_or_fitted_audit_sources(condition,windows,updates):
    with pytest.raises(AssertionError):audit_start_for(condition,windows,updates)


def test_unregistered_source_condition_rejected():
    with pytest.raises(ValueError):audit_start_for('selected-by-loss',951172)
