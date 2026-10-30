"""Numerical/parser fixtures only; never a generated research corpus."""
import pytest
from compute_trainium.authored_direct_arithmetic import direct_equalities


@pytest.mark.parametrize('text,expected', [
    ('The result is 2+3*4 = 14 now.', '14'),
    ('Evaluate (7-2)*3 = 15', '15'),
    ('We have 1/2+1/4 = 3/4', '3/4'),
    ('Compute 2−3×4 = −10', '−10'),
    ('Then 2+3*4=14=7*2.', '14'),
])
def test_returns_existing_answer_without_reformatting(text, expected):
    row, = direct_equalities(text)
    assert text[row['target_start']:row['target_end']] == expected
    assert row['operations'] >= 2
    assert set(row) == {'premise_start','premise_end','target_start','target_end','operations'}


@pytest.mark.parametrize('text', [
    '2+3*4=13', 'x2+3*4=14', 'x 2+3*4=14', 'sin 2+3*4=14',
    '2+3*4=14x', '2+3*4=14+1', '2+3=5', '2+3-2=3',
    '2+3*4==14', '2+3*4>=14', '2**10000000+1=5',
    '2/0+1=3', 'a_2+3*4=14', '2+3*4=14.5',
])
def test_invalid_symbolic_copyable_or_incomplete_equalities_excluded(text):
    assert list(direct_equalities(text)) == []
