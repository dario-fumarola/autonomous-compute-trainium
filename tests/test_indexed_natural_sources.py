import json
import numpy as np
import pytest
import torch
from compute_trainium.indexed_natural_sources import IndexedNaturalSources,RECORD_DTYPE,DOMAINS

def fixture(tmp_path,lengths=(3,9,1,17,4),context=8):
    source=tmp_path/'source';source.mkdir();out=tmp_path/'index';out.mkdir()
    arrays={d:np.arange(40,dtype=np.uint16)+10+100*i for i,d in enumerate(DOMAINS)}
    for d,a in arrays.items():a.tofile(source/(d+'.u16'))
    records=[];flat=[];domain=[];position=0
    for i,n in enumerate(lengths):
        code=i%3+1;offset=i
        records.append((position,offset,n,code));flat += [1]+arrays[DOMAINS[code-1]][offset:offset+n].tolist()+[2]
        domain += [0]+[code]*n+[0];position+=n+2
    windows=(position+context-2)//context
    np.save(out/'PRIVATE-documents.npy',np.array(records,dtype=RECORD_DTYPE))
    (out/'report.json').write_text(json.dumps(dict(status='PASS_INDEXED_NATURAL_SOURCES',source_relative='source',context=context,windows=windows,valid_positions=position)))
    pad=windows*context+1-position;flat += [2]*pad;domain += [0]*pad
    return IndexedNaturalSources(out),np.array(flat),np.array(domain)

def test_every_window_boundary_and_target_count(tmp_path):
    source,flat,domain=fixture(tmp_path)
    for i in range(source.windows):
        a,b=source.window(i);left=i*source.context
        np.testing.assert_array_equal(a,flat[left:left+source.context+1]);np.testing.assert_array_equal(b,domain[left:left+source.context+1])
        assert source.target_count(i,1)==np.count_nonzero(b[1:])
    for p in range(len(flat)+1):assert source.prefix_targets(p)==np.count_nonzero(domain[:p])
    assert source.target_count(0,source.windows)==sum(source.records['length'])

def test_real_targets_once_and_batch_order(tmp_path):
    source,flat,domain=fixture(tmp_path)
    x,y,m=source.batch(list(range(source.windows)))
    assert torch.equal(y.flatten(),torch.tensor(flat[1:]))
    assert torch.equal(m.flatten(),torch.tensor(domain[1:]!=0))
    assert m.sum()==sum(source.records['length'])
    reverse=source.batch(list(reversed(range(source.windows))))
    for a,b in zip((x,y,m),reverse):assert torch.equal(a.flip(0),b)
    for i in range(source.windows-1):assert x[i+1,0]==y[i,-1]

def test_range_and_schema_rejection(tmp_path):
    source,_,_=fixture(tmp_path)
    with pytest.raises(IndexError):source.window(source.windows)
    with pytest.raises(ValueError):source.target_count(1,source.windows)
    with pytest.raises(ValueError):source.batch([])
    rows=source.records.copy();rows['start'][1]+=1;np.save(source.root/'PRIVATE-documents.npy',rows)
    with pytest.raises(ValueError):IndexedNaturalSources(source.root)
