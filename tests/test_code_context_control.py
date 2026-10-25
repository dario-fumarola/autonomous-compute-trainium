import copy
import pytest

from compute_trainium.code_context_control import compare_trained_control


def test_trained_control_uses_supported_exit_pairing_and_original_band_weights():
    controls=[];candidate=[]
    for band in ['129-256','257-512','513-900']:
        for i in range(8):
            info={'id':band+str(i),'group':band+str(i),'band':band,'real_targets':10}
            controls.append({**info,'nll_sum':{'full':20.,'short':30.}})
            candidate.append({**info,'nll_sum':{'full1':19.,'full2':18.,'short1':31.,'short2':29.}})
    r=compare_trained_control(controls,candidate,draws=200)
    assert r['candidate_gain_full']['gain_nats']==pytest.approx(.2)
    assert r['candidate_gain_short']['gain_nats']==pytest.approx(.1)
    assert r['cross_model_context_interaction']['gain_nats']==pytest.approx(.1)
    assert r['identifier_targets']==24 and r['real_target_tokens']==240
    changed=copy.deepcopy(controls);changed[0]['group']='wrong-repository'
    with pytest.raises(ValueError):compare_trained_control(changed,candidate,draws=200)
    with pytest.raises(ValueError):compare_trained_control(controls+[controls[0]],candidate,draws=200)
