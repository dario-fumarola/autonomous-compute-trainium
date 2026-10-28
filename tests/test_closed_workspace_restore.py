import gzip
import hashlib
import json
import pytest
from scripts import preserve_closed_workspace_checkpoints as module


def fixture(root,monkeypatch):
    original=b'original checkpoint archive bytes'*32
    archive=root/'cold.gz';archive.write_bytes(gzip.compress(original))
    out=root/module.IDENTITY;out.mkdir()
    row=dict(relative='state.zip',archive_relative='cold.gz',bytes=len(original),sha256=hashlib.sha256(original).hexdigest(),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps({'files':[row]}))
    monkeypatch.setattr(module,'managed_running',lambda:[])
    monkeypatch.setattr(module.shutil,'disk_usage',lambda p:type('D',(),{'free':100*1024**3})())
    return original,archive


def test_materialized_restore_preserves_original_bytes_and_rejects_overwrite(tmp_path,monkeypatch):
    original,archive=fixture(tmp_path,monkeypatch)
    module.restore(tmp_path,'state.zip')
    assert (tmp_path/'state.zip').read_bytes()==original
    with pytest.raises(RuntimeError,match='non-overwriting'):module.restore(tmp_path,'state.zip')


def test_corrupt_archive_is_rejected_before_creating_restored_state(tmp_path,monkeypatch):
    original,archive=fixture(tmp_path,monkeypatch)
    archive.write_bytes(archive.read_bytes()+b'corruption')
    with pytest.raises(RuntimeError,match='intact'):module.restore(tmp_path,'state.zip')
    assert not (tmp_path/'state.zip').exists()
