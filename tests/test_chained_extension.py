"""Tiny software metadata fixtures; no generated research examples."""
import json
import sqlite3

import pytest
import numpy as np

from compute_trainium.training import file_sha256
from scripts.prepare_chained_extension import lineage,old_raw_paths,source_key
from scripts.prepare_future_extension import prior_identities
from scripts.audit_chained_extension import audit_target_prefixes


def generation(root,identity,parent=None):
    root.mkdir();source={'identity':identity}
    manifest={'status':'PASS_FROZEN_SOFTWARE_FIXTURE','sources':{}}
    if parent:
        source['parent_source_sha256']=file_sha256(parent/'source.json')
        manifest['parent_manifest_sha256']=file_sha256(parent/'manifest.json')
    (root/'source.json').write_text(json.dumps(source));(root/'manifest.json').write_text(json.dumps(manifest))
    path=root/('extension/documents.sqlite' if parent else 'documents.sqlite');path.parent.mkdir(exist_ok=True)
    db=sqlite3.connect(path);db.execute('create table documents(id text,domain text,source_group text,split text)')
    db.execute('insert into documents values(?,?,?,?)',(identity,'web',identity,'train'));db.commit();db.close()
    return root


def test_all_three_generations_are_required_in_order(tmp_path):
    first=generation(tmp_path/'a','first');second=generation(tmp_path/'b','second',first)
    third=generation(tmp_path/'c','third',second)
    manifests,databases=lineage([first,second,third])
    ids,groups,domains=prior_identities(databases,[])
    assert ids=={'first','second','third'} and not groups and domains['web']==ids
    for invalid in [[first,third],[first,third,second],[first,second,second]]:
        with pytest.raises(ValueError):lineage(invalid)
    (second/'source.json').write_text('{}')
    with pytest.raises((ValueError,KeyError)):lineage([first,second,third])


def test_raw_receipts_distinguish_identical_basename_in_different_crawls(tmp_path):
    root=tmp_path/'generation';root.mkdir();(root/'raw').mkdir()
    tokenizer=tmp_path/'tokenizer.json';tokenizer.write_text('{}')
    paths=[root/'raw/web-0.parquet',root/'raw/web-1.parquet']
    receipts=[]
    for index,path in enumerate(paths):
        path.write_bytes(bytes([index]))
        receipts.append({'repository':'fixture','revision':'fixed','file':f'crawl{index}/train-00000.parquet','sha256':file_sha256(path)})
    mapping=old_raw_paths([root],[{'sources':{'web':receipts}}],tokenizer)
    assert len(mapping)==2 and mapping[source_key('web',receipts[1])][0]==paths[1].resolve()
    paths[1].write_bytes(b'changed')
    with pytest.raises(ValueError):old_raw_paths([root],[{'sources':{'web':receipts}}],tokenizer)


def test_raw_map_supports_a_later_nonlegacy_generation(tmp_path):
    root=tmp_path/'generation';root.mkdir();raw=tmp_path/'private-source';raw.write_bytes(b'fixture')
    receipt={'repository':'fixture','revision':'fixed','file':'crawl/shard.parquet','sha256':file_sha256(raw)}
    (root/'raw-source-map.json').write_text(json.dumps([{'domain':'web','path':str(raw),'receipt':receipt}]))
    mapping=old_raw_paths([root],[{'sources':{'web':[receipt]}}],tmp_path/'tokenizer.json')
    assert mapping[source_key('web',receipt)][0]==raw.resolve()


def test_full_prefix_audit_catches_interior_counts_and_invalid_mask_or_vocabulary():
    tokens=np.ones(3073,dtype=np.int32);domains=np.zeros(3073,dtype=np.uint8)
    domains[[1,1025,2049]]=[1,2,3]
    counts=np.array([[0,0,0],[1,0,0],[1,1,0],[1,1,1]],dtype=np.int64)
    assert np.array_equal(audit_target_prefixes(tokens,domains,counts,3),[1,1,1])
    corrupted=counts.copy();corrupted[1]=[1,1,0]
    with pytest.raises(AssertionError,match='prefix'):audit_target_prefixes(tokens,domains,corrupted,3)
    domains[1500]=4
    with pytest.raises(AssertionError,match='mask'):audit_target_prefixes(tokens,domains,counts,3)
    domains[1500]=0;tokens[1600]=32000
    with pytest.raises(AssertionError,match='vocabulary'):audit_target_prefixes(tokens,domains,counts,3)
