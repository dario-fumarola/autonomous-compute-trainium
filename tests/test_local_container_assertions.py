"""Static-verifier fixtures only, never research examples."""
import pytest

from compute_trainium.local_container_assertions import verified_container_assertions


def test_authored_answer_span_and_changing_state():
    source = 'xs = ["a"]\nxs.append("b")\nxs.extend(["c", "d"])\nxs.pop()\nassert len(xs) == 3\n'
    row, = verified_container_assertions(source)
    assert source[row['target_start']:row['target_end']] == '3'
    assert row['operations'] == 3


@pytest.mark.parametrize('extra', [
    'other = xs', 'external(xs)', 'xs.sort()', 'xs = make_list()',
    'if flag:\n    xs.append("x")', 'xs.append(unknown)', 'xs.pop(99)',
    'xs.extend(*args)', 'xs.append(value="a")',
])
def test_unknown_effects_and_aliases_invalidate_state(extra):
    source = 'xs = []\nxs.append("a")\nxs.append("b")\n'+extra+'\nassert len(xs) == 2\n'
    assert verified_container_assertions(source) == []


def test_reject_wrong_answer_and_do_not_follow_failed_assertion():
    source = 'xs = []\nxs.append("a")\nxs.append("b")\nassert len(xs) == 7\nassert len(xs) == 2\n'
    assert verified_container_assertions(source) == []


@pytest.mark.parametrize('source', [
    'xs = [2]\nxs.append("a")\nxs.pop()\nxs.append("b")\nassert len(xs) == 2',
    'len = custom\nxs = []\nxs.append("a")\nxs.append("b")\nassert len(xs) == 2',
    'def f(len):\n    xs = []\n    xs.append("a")\n    xs.append("b")\n    assert len(xs) == 2',
])
def test_copy_and_rebound_builtin_excluded(source):
    assert verified_container_assertions(source) == []


def test_function_scope_literals_without_execution():
    source = 'def f():\n    xs = []\n    xs.append("a")\n    xs.append("b")\n    assert len(xs) == 2\n'
    assert len(verified_container_assertions(source)) == 1


@pytest.mark.parametrize('prefix', ['from module import len', 'from module import *', 'import module as len', 'builtins.len = custom', 'def len(x):\n    return 2'])
def test_explicit_import_or_attribute_shadowing_excluded(prefix):
    source = prefix+'\nxs = []\nxs.append("a")\nxs.append("b")\nassert len(xs) == 2\n'
    assert verified_container_assertions(source) == []
