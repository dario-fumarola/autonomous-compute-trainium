"""Parser software fixtures only, never generated research examples."""
import pytest
from compute_trainium.authored_doctest_arithmetic import verified_doctest_arithmetic


def documented(body):return 'def example():\n    """Example.\n'+''.join('    '+s+'\n' for s in body.splitlines())+'    """\n    pass\n'


def test_returns_only_original_authored_span_with_two_operations():
    s=documented('>>> total = 7 * 3\n>>> total - 2\n19')
    rows=verified_doctest_arithmetic(s);assert len(rows)==1
    r=rows[0];assert s[r['target_start']:r['target_end']]=='19'
    assert r['operations']==2 and 'total = 7 * 3' in s[r['premise_start']:r['premise_end']]


@pytest.mark.parametrize('body',[
    '>>> total = 7 * 3\n>>> total - 2\n18',
    '>>> total = unknown()\n>>> total - 2\n19',
    '>>> total = 7 * 3\n>>> unknown()\n>>> total - 2\n19',
    '>>> 21 - 2\n19',
    '>>> 7 * 3 - 2 + 19 - 19\n19',
    '>>> total = 7 * 3\n>>> total - 2\n+19',
    '>>> import os; os.system("false")\n>>> total - 2\n19',
    '>>> total = 7 * 3\n>>> total / 2\n10',
])
def test_wrong_copyable_unknown_or_noncanonical_answers_are_excluded(body):
    assert verified_doctest_arithmetic(documented(body))==[]


def test_scope_and_source_encoding_are_preserved():
    a=documented('>>> total = 7 * 3\n>>> total - 2\n19').replace('Example.','Résumé.')
    b='def another():\n    """>>> total - 2\n19\n"""\n'
    rows=verified_doctest_arithmetic(a+b);assert len(rows)==1
    assert (a+b)[rows[0]['target_start']:rows[0]['target_end']]=='19'
    assert verified_doctest_arithmetic('message = """>>> 7 * 3 - 2\n19\n"""\n')==[]
