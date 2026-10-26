"""Source preservation software fixtures;no model examples."""
import hashlib
import json
import pytest

from compute_trainium.closed_source_storage import source_record,source_actions,share_source


def fixture(root,name,text='x = 1\n'):
    p=root/'snapshots'/name;p.mkdir(parents=True);source=p/'model.py';source.write_text(text)
    h=hashlib.sha256(source.read_bytes()).hexdigest()
    (p/'launch.json').write_text(json.dumps(dict(source_sha256={'model.py':h})))
    (p/'closure.json').write_text('{"exit_code":0}')
    return p,source_record(root,p,'model.py',h)


def test_closed_sources_preserve_paths_hashes_and_execution_receipts(tmp_path):
    a,left=fixture(tmp_path,'a');b,right=fixture(tmp_path,'b');launch=(b/'launch.json').read_bytes()
    share_source(tmp_path,source_actions([left,right])[0])
    assert (a/'model.py').stat().st_ino==(b/'model.py').stat().st_ino
    assert (a/'model.py').read_bytes()==(b/'model.py').read_bytes()==b'x = 1\n'
    assert (b/'launch.json').read_bytes()==launch


def test_changed_historical_source_is_rejected_before_any_link(tmp_path):
    a,left=fixture(tmp_path,'a');b,right=fixture(tmp_path,'b');inode=(b/'model.py').stat().st_ino
    (b/'model.py').write_text('changed = True\n')
    with pytest.raises(AssertionError):share_source(tmp_path,source_actions([left,right])[0])
    assert (b/'model.py').stat().st_ino==inode


def test_open_or_outside_snapshot_cannot_be_coalesced(tmp_path):
    a,row=fixture(tmp_path,'a');(a/'closure.json').unlink()
    with pytest.raises(ValueError):source_record(tmp_path,a,'model.py',row['sha256'])
