import pytest

from compute_trainium import owned_native_compact as compact


def test_compact_cap_requires_ownership_and_verifies_memory_and_swap(monkeypatch):
    calls=[]
    state={'Config':{'Labels':{'ac-trainium-managed':'1','ac-trainium-owner':'mine'}},
           'HostConfig':{'Memory':128*1024**3,'MemorySwap':128*1024**3}}
    monkeypatch.setattr(compact,'create_worker',lambda *args:'container')
    monkeypatch.setattr(compact,'metadata',lambda *_:state)
    monkeypatch.setattr(compact.subprocess,'run',lambda command,**kwargs:calls.append(command))
    monkeypatch.setattr(compact,'cancel_worker_container',lambda *args:calls.append(args))
    assert compact.create_compact_worker('/root','mine',1)=='container'
    assert calls==[['docker','update','--memory','128g','--memory-swap','128g','container']]
    state['HostConfig']['MemorySwap']=160*1024**3
    with pytest.raises(ValueError):compact.create_compact_worker('/root','mine',1)
    assert calls[-1]==('container','mine','compact_worker_setup_failed')
    calls.clear();state['Config']['Labels']['ac-trainium-owner']='someone-else'
    with pytest.raises(ValueError):compact.create_compact_worker('/root','mine',1)
    assert not any(isinstance(c,list) for c in calls)
