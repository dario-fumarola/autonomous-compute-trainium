import pytest
from compute_trainium.low_depth_gradient_probe import validate_probe_state
from compute_trainium.parcae_scientific_state import signature


def test_probe_rejects_mixed_source_case_and_qualification_identity():
    parent={'purpose':'natural_low_depth_scientific_pilot','initial_states':{'i':{'sha256':'i'}},
        'model':{},'data':{'packing_fingerprint':'p','corpus_sha256':'s'},'accumulation':16,
        'precision':'fp16','routing_seed':1,'cases':{'c':{'initial_state':'i','arm':'poisson2',
        'seed':11,'state_seed':12,'depth_table_sha256':'d','total_updates':576}}}
    saved={'purpose':parent['purpose'],'qualification_only':False,'case_id':'c',
        'signature':signature('plan',parent,parent['cases']['c']),'cursor':20*512,'update':20}
    assert validate_probe_state(saved,'plan',parent,'c') is parent['cases']['c']
    for change in [{'qualification_only':True},{'cursor':0},{'case_id':'other'}]:
        with pytest.raises(ValueError):validate_probe_state({**saved,**change},'plan',parent,'c')
