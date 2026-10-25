import copy
import pytest

from compute_trainium.future_span_study_plan import CELLS,RATES,measured_budget,select_rates


def hardware():
    return {'status':'PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION','records':[
        {'case':{'identity':cell},'native':{'warm':{'warm_updates':100,'warm_seconds':80.,
            'wall_seconds':180.,'warm_targets_per_second':650000.}}} for cell in CELLS]}


def test_measured_budget_rejects_unqualified_incomplete_or_missing_actual_updates():
    data=hardware()
    assert measured_budget(data,stage='tune')['fits_registered_cap']
    assert measured_budget(data,stage='main')['fits_registered_cap']
    for field,value in [('status','RUNNING'),('records',data['records'][:-1])]:
        bad=copy.deepcopy(data);bad[field]=value
        with pytest.raises(ValueError):measured_budget(bad,stage='tune')
    bad=copy.deepcopy(data);bad['records'][0]['native']['warm']['warm_updates']=12
    with pytest.raises(ValueError):measured_budget(bad,stage='tune')
    slow=copy.deepcopy(data)
    for row in slow['records']:row['native']['warm'].update(warm_seconds=10000.,wall_seconds=11000.)
    assert not measured_budget(slow,stage='tune')['fits_registered_cap']


def test_selection_requires_all_four_equal_grids_and_deterministic_ties():
    rows=[{'cell':cell,'case_id':cell+str(lr),'peak_lr':lr,'metrics':{'weighted_nll':3. if lr!=.008 else 4.}}
          for cell in CELLS for lr in RATES]
    result=select_rates(rows)
    assert all(r['peak_lr']==.016 and not r['boundary_winner'] for r in result.values())
    with pytest.raises(ValueError):select_rates(rows[:-1])
    with pytest.raises(ValueError):select_rates(rows+[copy.deepcopy(rows[0])])
