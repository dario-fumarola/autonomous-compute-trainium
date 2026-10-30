"""Syntax fixtures only; no generated program enters research data."""
from scripts.inventory_authored_code_targets import inventory


def run(text):
    return inventory(text, [(i,i+1) for i in range(len(text))])


def test_counts_nested_control_flow_without_claiming_correctness():
    text='def f(x):\n for i in range(3):\n  x+=i\n return x\ndef test():\n assert f(2)==999\n'
    c,rows=run(text)
    assert c['unverified_context_candidates']==1
    assert c['candidate_body_has_For']==1
    assert c['candidate_top_level_assertion']==0
    assert len(rows)==1  # deliberately wrong expected value: syntax is not validation


def test_definition_after_target_is_not_a_causal_candidate():
    c,rows=run('assert f(2)==5\ndef f(x):\n return x*2+1\n')
    assert c['integer_expected_named_call']==1
    assert c['unverified_context_candidates']==0
    assert rows==[]


def test_nonnumeric_argument_is_not_supported_inventory_candidate():
    c,rows=run('def f(x):\n return x+1\nassert f(other)==5\n')
    assert c['integer_expected_prior_local_definition']==1
    assert c['literal_arguments_prior_local_definition']==0
    assert not rows
