from compute_trainium.authored_doctest_inventory import inventory, literal_kind


def source(body):
    return '"""\n' + body + '\n"""\n'


def test_wrong_authored_answer_remains_explicitly_unverified():
    x = inventory(source('>>> a = 2 + 3\n>>> b = a * 4\n>>> b - 1\n999'))
    assert x['unverified_multistep_outputs'] == 1
    assert x['unverified_multistep_prior_variable'] == 1


def test_never_execute_calls_and_reset_unknown_environment():
    x = inventory(source('>>> a = 1 + 2\n>>> exec("raise RuntimeError")\n>>> a + 1\n4'))
    assert x['unsupported_or_context_open'] == 2
    assert x['unverified_multistep_outputs'] == 0


def test_no_scope_crossing_between_literals_or_blank_separated_examples():
    x = inventory(source('>>> a = 1 + 2') + source('>>> a * 3\n9'))
    assert x['unverified_multistep_outputs'] == 0
    y = inventory(source('>>> a = 1 + 2\n\n>>> a * 3\n9'))
    assert y['unverified_multistep_outputs'] == 0


def test_boolean_literals_and_builtin_shadowing():
    assert literal_kind('False\n') == 'boolean'
    assert literal_kind('-3\n') == 'integer'
    assert literal_kind('1 + 2\n') is None
    x = inventory(source('>>> x = 1 + 2\n>>> x > 2\nTrue'))
    assert x['unverified_boolean_closed_outputs'] == 1
    y = inventory(source('>>> len = 4\n>>> len([1, 2])\n2'))
    assert y['unsupported_or_context_open'] == 1


def test_unknown_attribute_and_import_are_not_closed_computation():
    x = inventory(source('>>> import math\n>>> math.sqrt(9)\n3'))
    assert x['unsupported_or_context_open'] == 2
