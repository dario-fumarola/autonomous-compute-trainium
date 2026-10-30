from scripts.summarize_primitive_operand_sizes import premise_ids


def test_bos_and_noninitial_windows_preserve_original_document_spans():
    assert premise_ids(dict(window_start=0,premise_tokens=[0,3],tokens=[1,11,12,13,14]))==[11,12,13]
    assert premise_ids(dict(window_start=5,premise_tokens=[6,9],tokens=[10,11,12,13,14]))==[11,12,13]


def test_calculator_literals_are_distinct_values_not_operand_occurrences():
    from compute_trainium.source_arithmetic import check
    parsed=check('8*8')
    assert parsed.operations==1 and len(parsed.literals)==1
    assert max(abs(v) for v in parsed.literals)==8
    signed=check('(-3)*(-5)')
    assert signed.operations==1 and len(signed.literals)==4
    assert max(abs(v) for v in signed.literals)==5
