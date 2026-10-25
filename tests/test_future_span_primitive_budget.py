from compute_trainium.future_span_study_plan import CELLS,measured_budget as original
from compute_trainium.future_span_primitive_budget import measured_budget


def test_measured_primitive_budget_retains_exposure_and_tune_cap_with_explicit_finite_main_amendment():
    # Software metadata fixture only,no devices or research examples.
    hardware={'status':'PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION','records':[
        {'case':{'identity':cell},'native':{'warm':{'warm_updates':100,
            'warm_seconds':160. if cell.startswith('recurrent') else 110.,
            'wall_seconds':324. if cell.startswith('recurrent') else 276.,
            'warm_targets_per_second':326000. if cell.startswith('recurrent') else 476000.}}}
        for cell in CELLS]}
    assert measured_budget(hardware,stage='tune')==original(hardware,stage='tune')
    old=original(hardware,stage='main');new=measured_budget(hardware,stage='main')
    assert not old['fits_registered_cap'] and new['fits_registered_cap']
    assert new['predicted_seconds_with_margin']==old['predicted_seconds_with_margin']
    assert old['execution_seconds_cap']==6*3600 and new['execution_seconds_cap']==8*3600
    assert new['supersedes_unexecuted_main_cap_seconds']==old['execution_seconds_cap']
