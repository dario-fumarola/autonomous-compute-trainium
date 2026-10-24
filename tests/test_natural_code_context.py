from compute_trainium.natural_code_context import unambiguous_local_uses,token_span,distance_band


def test_unicode_byte_columns_and_conditional_use_keep_direct_binding():
    # Parser/offset software fixture only; never a training/evaluation document.
    source='def f():\n    café = 3\n    if café:\n        return café\n'
    rows=unambiguous_local_uses(source)
    assert len(rows)==2
    assert all(source[r['target_start']:r['target_end']]=='café' for r in rows)
    assert all(source[r['definition_start']:r['definition_end']]=='café' for r in rows)


def test_rebinding_nested_scope_and_parameter_assignment_are_excluded():
    assert not unambiguous_local_uses('def f(x):\n    x=1\n    return x\n')
    assert not unambiguous_local_uses('def f():\n    x=1\n    x+=2\n    return x\n')
    source='def f():\n    x=1\n    def inner():\n        return x\n    return x\n'
    assert len(unambiguous_local_uses(source))==1
    assert not unambiguous_local_uses('def f():\n    if True:\n        x=1\n    return x\n')


def test_token_span_rejects_joined_punctuation_and_distance_edges():
    assert token_span(' a',[(0,2)],1,2)==(0,1)
    assert token_span('(a',[(0,2)],1,2) is None
    assert [distance_band(n) for n in [128,129,256,257,512,513,900,901]]==[None,'129-256','129-256','257-512','257-512','513-900','513-900',None]
