from pathlib import Path
from compute_trainium.training import file_sha256
from scripts.reclaim_closed_snapshot_bytecode import source_for_cache


def test_only_cache_with_present_exact_frozen_source_is_eligible(tmp_path):
    snapshot=tmp_path/'snapshot';snapshot.mkdir();source=snapshot/'module.py';source.write_text('value = 1\n')
    cache=snapshot/'__pycache__';cache.mkdir();bytecode=cache/'module.cpython-312.pyc';bytecode.write_bytes(b'numerical-file-fixture')
    assert source_for_cache(snapshot,bytecode,{'module.py':file_sha256(source)})==source
    assert source_for_cache(snapshot,bytecode,{}) is None
    model=snapshot/'checkpoint.pt';model.write_bytes(b'state-file-fixture')
    assert source_for_cache(snapshot,model,{'module.py':file_sha256(source)}) is None


def test_cache_symlink_cannot_escape_closed_snapshot(tmp_path):
    snapshot=tmp_path/'snapshot';snapshot.mkdir();outside=tmp_path/'outside';outside.mkdir()
    (outside/'module.cpython-312.pyc').write_bytes(b'fixture');(snapshot/'__pycache__').symlink_to(outside,target_is_directory=True)
    assert source_for_cache(snapshot,snapshot/'__pycache__/module.cpython-312.pyc',{}) is None
