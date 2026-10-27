import json

import pytest

from scripts.prepare_scale_main_storage import original_capture_sizes,main_state_allowance


def test_main_storage_accounts_raw_and_exact_retained_lengths_and_all_branches(tmp_path):
    root=tmp_path/'nvme';root.mkdir()
    for label,recurrent in [('fixed-two',True),('ordinary',False)]:
        name='scale-65536-'+label+'-numerical-v1';proof=root/name;proof.mkdir()
        (proof/'completion-verification.json').write_text(json.dumps({'status':'PASS_SCALE_65536_COMPLETE_NUMERICAL_QUALIFICATION'}))
        folder=tmp_path/'recurrence'/name/'warm';folder.mkdir(parents=True)
        for rank in range(64):
            (folder/f'rank{rank}-u100.pt.exact.json').write_text(json.dumps(dict(original_file=f'rank{rank}-u100.pt',original_bytes=100)))
            if recurrent:(folder/f'noise-u100-rank{rank}.pt').write_bytes(bytes(2))
        (folder/'gradient-u100.pt').write_bytes(bytes(30))
    report=main_state_allowance(root)
    assert report['complete_additional_state_bytes']==3*6400+30+128+6*6400+2*30
    assert report['margin_bytes']==4*1024**3
    file=tmp_path/'recurrence'/name/'warm/rank0-u100.pt';file.write_bytes(bytes(101))
    with pytest.raises(AssertionError):original_capture_sizes(file.parent)
