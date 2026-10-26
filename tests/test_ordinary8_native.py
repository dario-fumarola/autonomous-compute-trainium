import pytest
from scripts.qualify_ordinary8_managed import validate_phase


def plan():return dict(purpose='ordinary8_native_qualification_only',root_comm_ports={'owned-warm':17000},
                       phases={'owned-warm':dict(mode='warm')})


def test_exact_ordinary8_phase():validate_phase(plan(),'owned-warm',17000,arm='ordinary',mode='warm')


@pytest.mark.parametrize('arm,mode,port',[('fixed1','warm',17000),('ordinary','replay',17000),('ordinary','warm',17001)])
def test_reject_phase_or_architecture_change(arm,mode,port):
    with pytest.raises(ValueError):validate_phase(plan(),'owned-warm',port,arm=arm,mode=mode)
