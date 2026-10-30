"""Selection fixtures, not generated research data."""
from types import SimpleNamespace
from compute_trainium.direct_arithmetic_panel import expression_key,eligible_cases,choose_panel


def enc(text):
    return SimpleNamespace(offsets=[(i,i+1) for i in range(len(text))])


def test_canonical_literals_and_operation_shapes():
    assert expression_key('2 + 3*4')==expression_key('2+3 * 4')
    assert expression_key('2+3*4')!=expression_key('5+6*7')
    assert expression_key('2+3*4',blind=True)==expression_key('5+6*7',blind=True)


def test_answer_copy_and_symbolic_tail_are_excluded():
    text='Earlier answer 14. Now 2+3*4=14'
    assert not list(eligible_cases(text,enc(text)))
    text='total +2+3*4=14'
    assert not list(eligible_cases(text,enc(text)))
    text='Compute 2+3*4=14'
    row,=eligible_cases(text,enc(text))
    assert text[slice(*row['target_tokens'])]=='14'
    assert row['stratum']=='2'


def test_one_group_and_expression_without_replacement_selection():
    candidates=[dict(document_id=str(i),group=g,expression_key=e,target_tokens=[i,i+1],stratum=s)
                for i,(g,e,s) in enumerate([('a','x','4-6'),('a','y','3'),('b','x','3'),
                                           ('c','z','3'),('d','v','2')])]
    rows,counts,ready=choose_panel(candidates,quota=1,minimum=1)
    assert ready and counts=={'4-6':1,'3':1,'2':1}
    assert len({r['group'] for r in rows})==len({r['expression_key'] for r in rows})==3
    assert choose_panel(list(reversed(candidates)),quota=1,minimum=1)==(rows,counts,ready)
