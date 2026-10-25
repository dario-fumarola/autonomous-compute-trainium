from compute_trainium.source_problem_groups import literal_operator_group

def test_conservative_group_excludes_reordered_or_reparenthesized_case_values():
    assert literal_operator_group('1+2*3')==literal_operator_group('(3*2)+1')
    assert literal_operator_group('1+2*3')==literal_operator_group('(1+2)*3')
    assert literal_operator_group('1+2*3')!=literal_operator_group('1+2*4')
    assert literal_operator_group('1+1*3')!=literal_operator_group('1+3*3')
    assert literal_operator_group('-1+2*3')!=literal_operator_group('1+2*3')
