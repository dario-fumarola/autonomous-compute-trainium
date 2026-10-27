import hashlib
import os

import pytest

from scripts.closed_raw_cache_maintenance import validate_candidate,verify


def record(path):
    return dict(path=str(path),domain='web',receipt=dict(repository='HuggingFaceFW/fineweb-edu',
        revision='a'*40,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),file='data/public.parquet'))


def test_cache_scope_cannot_target_states_arrays_symlinks_or_unpinned_sources(tmp_path):
    cache=tmp_path/'raw';cache.mkdir();path=cache/'web-1.parquet';path.write_bytes(b'plain software cache fixture')
    item=record(path);assert validate_candidate(cache,item)[0]==path
    for name in ['model.pt','tokens.npy','documents.sqlite','web-1.parquet/../model.pt']:
        with pytest.raises(ValueError):validate_candidate(cache,{**item,'path':str(cache/name)})
    other=tmp_path/'outside.parquet';other.write_bytes(path.read_bytes());link=cache/'web-2.parquet';link.symlink_to(other)
    with pytest.raises(ValueError):validate_candidate(cache,{**item,'path':str(link)})
    bad={**item,'receipt':{**item['receipt'],'revision':'main'}}
    with pytest.raises(ValueError):validate_candidate(cache,bad)
    os.link(path,cache/'shared-copy')
    with pytest.raises(ValueError):validate_candidate(cache,item)


def test_manifest_retention_rejects_release_before_any_file_or_remote_action(tmp_path):
    import json
    source=tmp_path/'natural-append-17000m-v3';source.mkdir()
    (source/'failure.json').write_text(json.dumps(dict(status='STOPPED_CHAINED_DATA_PREPARATION')))
    capacity=tmp_path/'natural-append-capacity-v4';capacity.mkdir()
    (capacity/'manifest.json').write_text(json.dumps(dict(scope='Original raw sources remain indispensable private dependencies.')))
    with pytest.raises(ValueError,match='indispensable dependencies'):verify(tmp_path)
