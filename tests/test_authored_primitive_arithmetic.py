import pytest
from compute_trainium.authored_primitive_arithmetic import equalities, choose_panel


@pytest.mark.parametrize('text,op,answer', [
    ('Compute 8+9=17.', 'add', '17'),
    ('Compute 15-8=7.', 'subtract', '7'),
    ('Compute 7*8=56.', 'multiply', '56'),
    ('Compute 24/8=3.', 'divide', '3'),
    ('Compute (3-8)=-5.', 'subtract', '-5'),
])
def test_original_authored_spans_only(text, op, answer):
    rows = list(equalities(text)); assert len(rows) == 1
    row = rows[0]; assert row['stratum'] == op
    assert text[row['target_start']:row['target_end']] == answer


@pytest.mark.parametrize('text', [
    'Compute 8+9=18.', 'Compute 8+0=8.', 'Compute 2+3*4=14.',
    'Compute 3/2=1.5.', 'Compute 2**3=8.', 'Compute 1001+2=1003.',
    'Compute x + 2=7.', 'Compute sin (3+4)=7.', 'Compute 2+3=5x.',
])
def test_reject_wrong_copyable_composite_symbolic_and_out_of_scope(text):
    assert list(equalities(text)) == []


def test_selection_is_deterministic_source_and_expression_distinct():
    rows = [dict(document_id=str(i), group=str(i), expression_key=str(i),
                 target_tokens=[i, i+1], stratum=s)
            for i,s in enumerate(['divide','subtract','multiply','add']*3)]
    a, bins, ready = choose_panel(rows, quota=2, minimum=2)
    b, _, _ = choose_panel(list(reversed(rows)), quota=2, minimum=2)
    assert a == b and ready and set(bins.values()) == {2}
    assert len({r['group'] for r in a}) == 8
