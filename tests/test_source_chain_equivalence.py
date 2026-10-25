"""Extraction algorithm fixtures only;never a research corpus."""
from compute_trainium.source_arithmetic import original_chains
from compute_trainium.source_arithmetic_fast import original_chains as fast


def test_maximal_run_scan_preserves_source_offsets_and_first_eligible_order():
    fixtures=[
        'Given: 2*(3+4) = 2*7 = 14; end.',
        r'Work $$ (4\times9)-(10\times-2)=36-(-20)=36+20=56 $$',
        'Unsupported sqrt (2*(3+4)) = 2*7 = 14',
        'text\n'+('word 12 + 7\n'*200)+' 2*(3+4)=2*7=14.',
        r'Task \frac{8+4}{2}=12/2=6; then 2*(3+4)=2*7=14',
        '4 = 4 and -2+(3-3) = -2+0 = -2',
        'f(2*(3+4)) = 2*7 = 14',
    ]
    for text in fixtures:assert list(fast(text))==list(original_chains(text))
