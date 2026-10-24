"""Tiny accepted-record fixtures, never research examples."""
import sqlite3

import pytest

from scripts.finalize_capacity_reserve import capacity_selection,copy_prefix


def test_capacity_selection_preserves_whole_documents_and_the_fixed_mix():
    db=sqlite3.connect(':memory:');db.execute('create table documents(domain text,offset integer,length integer)')
    db.executemany('insert into documents values(?,?,?)',[
        ('web',0,100),('web',100,100),('web',200,100),('web',300,100),
        ('math',0,100),('math',100,100),('code',0,100)])
    counts,quotas,selected=capacity_selection(db)
    assert counts=={'web':400,'math':200,'code':100}
    assert quotas==selected=={'web':300,'math':100,'code':100}
    db.execute("update documents set length=95 where domain='code'")
    _,quotas,selected=capacity_selection(db)
    assert quotas=={'web':285,'math':95,'code':95}
    assert selected=={'web':300,'math':100,'code':95}
    db.execute("update documents set length=100 where domain='code'")
    db.execute("update documents set offset=101 where domain='math' and offset=100")
    db.execute("update documents set length=99 where domain='math' and offset=0")
    with pytest.raises(ValueError):capacity_selection(db)


def test_prefix_copy_keeps_original_and_rejects_short_source(tmp_path):
    source=tmp_path/'original';source.write_bytes(bytes(range(30)))
    copy_prefix(source,tmp_path/'selected',20)
    assert source.read_bytes()==bytes(range(30)) and (tmp_path/'selected').read_bytes()==bytes(range(20))
    with pytest.raises(ValueError):copy_prefix(source,tmp_path/'short',31)
