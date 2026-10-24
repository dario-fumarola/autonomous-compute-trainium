"""Numerical streaming/identity fixtures; none are research examples."""
import itertools
import json
import shutil

import numpy as np
import pytest

from compute_trainium.heldout_screen import PriorNeighborhoods, panel_role
from compute_trainium.training import file_sha256
from scripts import prepare_confirmation_reserve as reserve


def test_sequential_batches_bound_readers_and_preserve_raw_row_locations(monkeypatch):
    active=set()
    maximum=0
    def rows(path):
        nonlocal maximum
        active.add(path);maximum=max(maximum,len(active))
        try:
            for i in range(300):yield {'source':path,'raw_row':i}
        finally:active.remove(path)
    monkeypatch.setattr(reserve,'rows',rows)
    batches=list(reserve.sequential_batches('web',['first','second','third']))
    assert maximum==1 and not active
    assert [(shard,start,len(values)) for _,shard,start,values in batches]==[
        (0,0,256),(0,256,44),(1,0,256),(1,256,44),(2,0,256),(2,256,44)]
    for domain,shard,start,values in batches:
        assert domain=='web'
        assert [v['raw_row'] for v in values]==list(range(start,start+len(values)))
    pending=reserve.sequential_batches('web',['first','second'])
    next(pending);pending.close()
    assert not active


def test_prior_scan_order_cannot_change_acceptance_when_rejected_work_is_skipped():
    observations=[(0,(0,i),False) for i in range(257)]
    observations += [(1,(0,i),i==20) for i in range(40)]
    observations += [(2,(0,i),False) for i in range(20)]
    results=[]
    for sequence in [observations,list(reversed(observations)),observations[::2]+observations[1::2]]:
        summary=PriorNeighborhoods(3)
        for candidate,identity,near in sequence:
            if summary.reason(candidate) is None:summary.observe(candidate,identity,near)
        results.append([summary.reason(i) is not None for i in range(3)])
        assert len(summary.neighbors[2])==20
    assert results==[[True,True,False]]*3


def test_candidate_pool_resume_requires_pinned_inputs_and_bytes(tmp_path):
    origin=tmp_path/'origin';origin.mkdir();(origin/'index').mkdir()
    plan={key:['same'] for key in ['generation_manifest_sha256','prior_database_sha256','protected_panel_sha256']}
    plan.update(tokenizer_sha256='same',selection='same fixed rule',real_target_quota_per_domain_role=250000,
                minimum_groups_per_domain_role=100,near_duplicate_rule='same duplicate rule',
                source_sha256={'scripts/prepare_confirmation_reserve.py':'previous-driver','helper.py':'same-helper'})
    (origin/'plan.json').write_text(json.dumps(plan))
    (origin/'raw-source-map.json').write_text('[]')
    doc={'id':'a'*64,'group':'c'*64,'domain':'web','role':panel_role('c'*64),'tokens':[11,12,13]}
    (origin/'private-candidate-pool.json').write_text(json.dumps([doc]))
    np.save(origin/'pool-signatures.npy',np.arange(32,dtype=np.uint64)[None])
    np.save(origin/'pool-sketches.npy',np.arange(256,dtype=np.uint64)[None])
    np.save(origin/'pool-sketch-lengths.npy',np.asarray([256],dtype=np.int16))
    np.save(origin/'index/ids-00.npy',np.asarray(['b'*64],dtype='S64'))
    np.save(origin/'index/domains-00.npy',np.asarray([1],dtype=np.uint8))
    # Partial coverage/rejection from the failed attempt must never be copied.
    np.save(origin/'index/found-00.npy',np.asarray([True]))
    np.save(origin/'pool-rejected.npy',np.asarray([True]))
    expected={'plan_sha256':file_sha256(origin/'plan.json'),
              'raw_source_map_sha256':file_sha256(origin/'raw-source-map.json'),
              'files':{name:file_sha256(origin/name) for name in reserve.POOL_FILES},
              'prior_index_files':{name:file_sha256(origin/'index'/name) for name in ['ids-00.npy','domains-00.npy']}}
    newplan={**plan,'source_sha256':{**plan['source_sha256'],'scripts/prepare_confirmation_reserve.py':'new-driver'}}
    def output(name):
        root=tmp_path/name;root.mkdir();(root/'index').mkdir()
        for entry in expected['prior_index_files']:shutil.copyfile(origin/'index'/entry,root/'index'/entry)
        shutil.copyfile(origin/'raw-source-map.json',root/'raw-source-map.json')
        np.save(root/'index/found-00.npy',np.asarray([False]))
        return root
    target=output('target')
    pool,signatures,sketches=reserve.restore_candidate_pool(origin,expected,newplan,target,set())
    assert pool==[doc] and signatures.shape==(1,32) and len(sketches[0])==256
    assert not np.load(target/'index/found-00.npy').any()
    assert not (target/'pool-rejected.npy').exists()
    with pytest.raises(AssertionError,match='inputs/rule changed'):
        reserve.restore_candidate_pool(origin,expected,{**newplan,'tokenizer_sha256':'different'},output('wrong-input'),set())
    (origin/'private-candidate-pool.json').write_text('[]')
    with pytest.raises(AssertionError,match='candidate pool changed'):
        reserve.restore_candidate_pool(origin,expected,newplan,output('changed-bytes'),set())
