"""Syntax-only software fixtures; never training/evaluation examples."""
import pytest

from compute_trainium.authored_code_assertions import Unsupported
from scripts.census_train_code_assertions import census


def test_syntax_categories_are_not_computation_certificates():
    row = census('assert f(3) == 9\nassert (a + b) * c == 12, "message"\nself.assertEqual(x, -7)\n')
    assert row['assert_statements'] == 2
    assert row['assert_equality_integer_rhs'] == 2
    assert row['integer_assert_lhs_has_call'] == 1
    assert row['integer_assert_lhs_two_arithmetic_nodes'] == 1
    assert row['assert_integer_rhs_with_message'] == 1
    assert row['assertEqual_integer_operand'] == 1


def test_booleans_and_non_equality_are_not_integer_equalities():
    row = census('assert a == True\nassert a != 3\nassert a == b == 7\nself.assertEqual(a, False)\n')
    assert row['assert_equality_integer_rhs'] == 0
    assert row['assertEqual_integer_operand'] == 0


def test_strings_are_not_executed_and_bounds_hold():
    row = census('""">>> assert 1 == 2"""\n')
    assert row['string_constants_containing_doctest_prompt'] == 1
    assert row['assert_statements'] == 0
    with pytest.raises(Unsupported):
        census(' ' * 100001)
