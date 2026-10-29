"""Parser fixtures only: no generated program enters a research corpus."""
import pytest

from compute_trainium.authored_code_assertions import verified_assertions, Unsupported


def test_authored_answer_span_and_unicode_offsets():
    source = '# λ\ndef compute(a, b):\n    c = a * 3\n    c -= b\n    return c + 1\nassert compute(7, 2) == 20\n'
    rows = verified_assertions(source)
    assert len(rows) == 1
    r = rows[0]
    assert source[r['target_start']:r['target_end']] == '20'
    assert source[r['call_start']:r['call_end']] == 'compute(7, 2)'
    assert source[r['function_start']:r['function_end']].startswith('def compute')
    assert r['operations'] == 3
    assert set(r) == {'target_start', 'target_end', 'function_start', 'function_end',
                      'call_start', 'call_end', 'operations', 'arguments'}


@pytest.mark.parametrize('source', [
    'def f(x):\n return x*2+1\nassert f(2)==6',  # incorrect authored answer
    'def f(x):\n return x*2\nassert f(3)==6',  # only one operation
    'def f(x):\n return x+4-1\nassert f(1)==4',  # expected value is a literal
    'def f(x):\n return eval(x)\nassert f(2)==6',
    'def f(x):\n return x.real+2\nassert f(2)==4',
    'def f(x):\n return x+global_value\nassert f(2)==6',
    '@decorate\ndef f(x):\n return x*2+1\nassert f(2)==5',
    'def f(x=mutate()):\n return x*2+1\nassert f(2)==5',
    'def f(x):\n return x*2+1\nf=other\nassert f(2)==5',
    'def f(x):\n return x*2+1\nunknown()\nassert f(2)==5',
    'def f(x):\n return x*2+1\ndef g(*x: mutate()):\n return 1\nassert f(2)==5',
    'def f(x,y):\n return x*2+1\nassert f(2,5)==5',
    'def f(x):\n unused=5\n return x*2+1\nassert f(2)==5',
    'def f(x):\n return x*2+1\nassert f(True)==3',
    'def f(x):\n return x//0+1\nassert f(2)==5',
    'def f(x):\n return x**100000000+1\nassert f(2)==5',
    'def f(x):\n for i in range(3): x+=i\n return x+1\nassert f(2)==6',
])
def test_unknown_incorrect_unsafe_or_copyable_cases_are_not_candidates(source):
    assert verified_assertions(source) == []


def test_negative_floor_division_and_original_signed_target():
    source = 'def f(x):\n return (x-8)//3\nassert f(1)==-3'
    row, = verified_assertions(source)
    assert source[row['target_start']:row['target_end']] == '-3'


def test_incorrect_assertion_does_not_validate_later_program_output():
    source = 'def f(x):\n return x*2+1\nassert f(2)==7\nassert f(3)==7'
    assert verified_assertions(source) == []


def test_docstrings_are_not_executed_or_mined_as_actual_assertions():
    source = '"""assert f(2)==5"""\ndef f(x):\n """arithmetic"""\n return x*2+1\nassert f(2)==5'
    assert len(verified_assertions(source)) == 1


def test_source_size_is_bounded():
    with pytest.raises(Unsupported):
        verified_assertions(' '*100001)
