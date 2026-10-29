"""Software fixtures only; these strings are never research examples."""
import pytest

from compute_trainium.local_arithmetic_assertions import verified_local_assertions


def test_local_multistep_answer_is_an_original_source_substring():
    text = 'def test_example():\n    x = 7 + 2\n    y = x * 3\n    assert y == 27\n'
    rows = verified_local_assertions(text)
    assert len(rows) == 1 and rows[0]['operations'] == 2
    row = rows[0]
    assert text[row['target_start']:row['target_end']] == '27'
    assert text[row['premise_start']:].startswith('x = 7 + 2')
    assert '27' not in text[row['premise_start']:row['premise_end']]


@pytest.mark.parametrize('prefix', ['unknown()', 'x = unknown()', 'if flag:\n        x = 99', 'import example'])
def test_unknown_statement_cannot_preserve_assumed_bindings(prefix):
    text = 'def test_example():\n    x = 7 + 2\n    '+prefix+'\n    assert x * 3 == 27\n'
    assert verified_local_assertions(text) == []


def test_unknown_parameter_and_future_definition_are_unavailable():
    assert verified_local_assertions('def f(x):\n    assert (x+2)*3 == 27\n    x=7\n') == []


def test_scopes_do_not_share_local_bindings():
    text = 'x=7+2\ndef f():\n    assert x*3 == 27\n'
    assert verified_local_assertions(text) == []


def test_expected_answer_copy_and_single_operation_are_excluded():
    assert verified_local_assertions('x=7+2\nassert x == 9\n') == []
    assert verified_local_assertions('x=7+2 # expected 27\nassert x*3 == 27\n') == []


def test_false_assertion_stops_later_candidates():
    assert verified_local_assertions('x=7+2\nassert x*3 == 28\nassert x*4 == 36\n') == []


def test_correct_class_method_and_negative_integer():
    text = 'class Test:\n    def test_x(self):\n        x = 3 - 8\n        assert x * 4 == -20\n'
    assert len(verified_local_assertions(text)) == 1


def test_float_call_and_zero_division_fail_closed_without_execution():
    assert verified_local_assertions('x=7+2\nassert open("never-open-this").read() == 27\n') == []
    assert verified_local_assertions('x=7.0+2\nassert x*3 == 27\n') == []
    assert verified_local_assertions('x=7+2\nassert x//0 == 27\n') == []
