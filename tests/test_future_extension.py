"""Software metadata/text fixtures only, never a research corpus."""
import hashlib
import json
import sqlite3

import pytest

from compute_trainium.data import content_hash, source_group, split_for_group
from scripts import prepare_future_extension as prep


def database(path, rows):
    db=sqlite3.connect(path)
    db.execute('create table documents(id text,domain text,source_group text,split text)')
    db.executemany('insert into documents values(?,?,?,?)',rows);db.commit();db.close()
    return path


def test_second_append_includes_both_generations_and_new_panels_in_exclusions(tmp_path):
    old=database(tmp_path/'a.db',[('a','web','train-group','train'),('b','math','held-group','dev')])
    extension=database(tmp_path/'b.db',[('c','code','train-code','train')])
    panel=tmp_path/'panel.json';panel.write_text(json.dumps([
        {'id':'d','domain':'code','group':'fresh-test','split':'test'}]))
    ids,groups,domains=prep.prior_identities([old,extension],[panel])
    assert ids=={'a','b','c','d'} and groups=={'held-group','fresh-test'}
    assert domains=={'web':{'a'},'math':{'b'},'code':{'c','d'}}
    panel.write_text(json.dumps([{'id':'e','domain':'web','group':'train-group','split':'test'}]))
    with pytest.raises(ValueError):prep.prior_identities([old,extension],[panel])
    panel.write_text(json.dumps([{'id':'e','domain':'web','group':'new','split':'train'}]))
    with pytest.raises(ValueError):prep.prior_identities([old,extension],[panel])


def test_reserved_group_is_rejected_even_if_its_original_hash_assigns_train(monkeypatch):
    urls=[]
    for index in range(100):
        url=f'https://fixture{index}.example/source'
        if split_for_group(source_group('web',url))=='train':urls.append(url)
        if len(urls)==2:break
    protected=hashlib.sha256(source_group('web',urls[0]).encode()).hexdigest()
    monkeypatch.setattr(prep,'PRIOR_IDS',{content_hash('old software fixture')},raising=False)
    monkeypatch.setattr(prep,'PROTECTED_GROUPS',{protected},raising=False)
    admitted=[]
    def record(rows):admitted.extend(rows);return [{'accepted':True}]
    monkeypatch.setattr(prep,'process_batch',record)
    results=prep.candidate_batch([
        ('web',0,{'text':'old software fixture','url':urls[1]}),
        ('web',0,{'text':'new protected fixture','url':urls[0]}),
        ('web',1,{'text':'new eligible fixture','url':urls[1]})])
    assert results==[{'reject':'prior_document'},{'reject':'held_out_source_group'},{'accepted':True}]
    assert len(admitted)==1 and admitted[0][1]==1


def test_raw_reconstruction_does_not_silently_omit_new_panel_identity(monkeypatch):
    text='A small software fixture for checking a raw content hash.'
    monkeypatch.setattr(prep,'PRIOR_IDS',{content_hash(text)},raising=False)
    rows=prep.inspect_old([('web',0,{'text':text}),('web',0,{'text':'unrelated fixture'})])
    assert len(rows)==1 and rows[0][0]==content_hash(text)
    assert rows[0][1].shape==(32,) and len(rows[0][2])>0
