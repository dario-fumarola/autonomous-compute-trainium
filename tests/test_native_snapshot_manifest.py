import pytest
from compute_trainium.native_snapshot_manifest import split_snapshot_sources


def test_reference_source_remains_frozen_without_requiring_uncopied_worker_path():
    sources={'scripts/capture.py':'a','compute_trainium/model.py':'b','configs/model.json':'c',
             'third_party/parcae_reference/muon_adamw.py':'d','tests/test_model.py':'e'}
    worker,host=split_snapshot_sources(sources)
    assert set(worker)=={'scripts/capture.py','compute_trainium/model.py','configs/model.json'}
    assert host['third_party/parcae_reference/muon_adamw.py']=='d'
    assert {**worker,**host}==sources
    for invalid in ['/outside.py','../outside.py','scripts/../outside.py']:
        with pytest.raises(ValueError):split_snapshot_sources({invalid:'x'})
