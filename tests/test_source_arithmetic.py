"""Parser software fixtures;never used as research training/evaluation data."""
from fractions import Fraction
import pytest
from compute_trainium.source_arithmetic import check,original_chains,Unsupported


def test_exact_decimal_fraction_and_tex_operator_verification():
    assert check('.1+.2').value==Fraction(3,10)
    assert check(r'\frac{2+4}{3} \times 5').value==10
    assert check('2^(1+2)').value==8
    assert check('−2+3').value==1
    for unsafe in ['__import__("os")','[1][0]','True','1/0','2^1000','x+3']:
        with pytest.raises((Unsupported,ValueError)):check(unsafe)


def test_only_original_equal_verified_spans_are_returned():
    source='Given: 2*(3+4) = 2*7 = 14; end.'
    records=list(original_chains(source));assert len(records)==1
    r=records[0]
    assert [source[a:b] for a,b in [r['expression'],*r['intermediate_steps'],r['conclusion']]]==['2*(3+4)','2*7','14']
    assert r['original_operations']==2 and r['step_operations']==[1]
    assert not list(original_chains('Given: 2*(3+4) = 2*7 = 15'))
    assert not list(original_chains('Given: 2*(3+4) = 14'))
    assert not list(original_chains('sqrt (2*(3+4)) = 2*7 = 14'))
    assert not list(original_chains('f(2*(3+4)) = 2*7 = 14'))


def test_signed_answer_copy_is_not_a_hidden_computation_example():
    assert not list(original_chains("Given: -2+(3-3) = -2+0 = -2"))
