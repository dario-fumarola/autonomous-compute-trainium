import copy
import pytest
from scripts.compare_interleaved_frontier import validate_exposure


def fixture():
    p=dict(seed=1,world=64,batch=2,weight_decay=.1,data_relative='source',data_report_sha256='a',
           evaluation_relative='dev',evaluation_report_sha256='b',audit_window_start=100)
    a=dict(real_targets=99,warmup_updates=1,decay_start=2,checkpoint_updates=[3],depth_schedule=[2,3,2])
    return p,a


def test_independent_shape_can_change_depth_but_not_source_exposure():
    p,a=fixture();q=copy.deepcopy(p);b=copy.deepcopy(a);b['depth_schedule']=[0,0,0]
    validate_exposure(p,q,a,b)


@pytest.mark.parametrize('key,value',[('seed',2),('data_report_sha256','c'),('evaluation_report_sha256','c'),('weight_decay',.8)])
def test_rejects_changed_pairing(key,value):
    p,a=fixture();q=copy.deepcopy(p);q[key]=value
    with pytest.raises(AssertionError):validate_exposure(p,q,a,a)


@pytest.mark.parametrize('key,value',[('real_targets',100),('decay_start',1),('checkpoint_updates',[2]),('depth_schedule',[0,0])])
def test_rejects_changed_endpoint_or_schedule(key,value):
    p,a=fixture();b=copy.deepcopy(a);b[key]=value
    with pytest.raises(AssertionError):validate_exposure(p,p,a,b)
