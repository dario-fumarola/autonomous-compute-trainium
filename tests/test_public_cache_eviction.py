import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import reclaim_public_download_cache as module


@pytest.mark.parametrize('corrupt',[False,True])
def test_only_verified_public_cache_is_evicted_and_bad_redownload_preserves_original(tmp_path,monkeypatch,corrupt):
    root=tmp_path/'root';root.mkdir();source=root/'natural-append-capacity-v4';source.mkdir()
    raw=root/'natural-old/raw';raw.mkdir(parents=True);file=raw/'sample.parquet';file.write_bytes(b'original bytes')
    receipt=dict(repository='fixture/public',revision='1'*40,file='sample.parquet',sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (source/'raw-source-map.json').write_text(json.dumps([dict(path=str(file),domain='web',receipt=receipt)]))
    model=root/'model.pt';model.write_bytes(b'private model remains')
    monkeypatch.setattr(module,'PUBLIC',tmp_path/'public')
    monkeypatch.setattr(module,'managed_running',lambda:[])
    monkeypatch.setattr(module.shutil,'disk_usage',lambda p:SimpleNamespace(free=44*1024**3))
    monkeypatch.setattr(module.requests,'head',lambda *a,**kw:SimpleNamespace(status_code=302,headers={'x-linked-etag':receipt['sha256']}))
    class Response:
        status_code=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,n):yield b'bad' if corrupt else b'original bytes'
    monkeypatch.setattr(module.requests,'get',lambda *a,**kw:Response())
    module.main(root)
    if corrupt:
        with pytest.raises(AssertionError):module.main(root,apply=True)
        assert file.read_bytes()==b'original bytes'
    else:
        module.main(root,apply=True);assert not file.exists()
        module.main(root,restore=str(file.relative_to(root)))
        assert file.read_bytes()==b'original bytes'
    assert model.read_bytes()==b'private model remains'
