import json
import subprocess

import pytest

from compute_trainium import owned_native_all_chips as target


def test_exclusive_claim_rejects_either_existing_half(tmp_path,monkeypatch):
    env=tmp_path/'isolated-native-env-v1';env.mkdir()
    (env/'parent-private.json').write_text(json.dumps(dict(HostConfig=dict(Devices=[]),Mounts=[])))
    (env/'setup-private.json').write_text(json.dumps(dict(image='software-test')))
    monkeypatch.setattr(target,'managed_running',lambda:['already-owned'])
    monkeypatch.setattr(target.subprocess,'run',lambda *a,**k:pytest.fail('Must reject before Docker mutation'))
    with pytest.raises(RuntimeError,match='every prior managed owner'):
        target.create_all_chip_worker(tmp_path,'software-test')


@pytest.mark.parametrize('memory',[128,200])
def test_exclusive_owner_has_no_half_and_uses_same_atomic_lock(tmp_path,monkeypatch,memory):
    env=tmp_path/'isolated-native-env-v1';env.mkdir()
    (env/'parent-private.json').write_text(json.dumps(dict(HostConfig=dict(Devices=[]),Mounts=[])))
    (env/'setup-private.json').write_text(json.dumps(dict(image='software-test')))
    monkeypatch.setattr(target,'managed_running',lambda:[])
    original_read=target.Path.read_text
    monkeypatch.setattr(target.Path,'read_text',lambda path,*args,**kwargs:
        'MemAvailable: 1073741824 kB\n' if str(path)=='/proc/meminfo' else original_read(path,*args,**kwargs))
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args,1 if args[1]=='inspect' else 0)
    monkeypatch.setattr(target.subprocess,'run',run)
    monkeypatch.setattr(target,'metadata',lambda name:dict(Config=dict(Labels={'ac-trainium-managed':'1','ac-trainium-owner':'software-test'})))
    assert target.create_all_chip_worker(tmp_path,'software-test',memory_gib=memory)=='ac-worker-software-test'
    assert (tmp_path/'native-half-reservations.lock').exists()
    args=calls[-1]
    assert 'ac-trainium-chips=all16' in args
    assert not any('ac-trainium-half' in v for v in args)
    assert args[args.index('--memory')+1]==str(memory)+'g'


def test_200_gib_still_requires_complete_512_gib_host_reserve(tmp_path,monkeypatch):
    env=tmp_path/'isolated-native-env-v1';env.mkdir()
    (env/'parent-private.json').write_text(json.dumps(dict(HostConfig=dict(Devices=[]),Mounts=[])))
    (env/'setup-private.json').write_text(json.dumps(dict(image='software-test')))
    monkeypatch.setattr(target,'managed_running',lambda:[]);original=target.Path.read_text
    monkeypatch.setattr(target.Path,'read_text',lambda p,*a,**k:'MemAvailable: '+str(710*1024**2)+' kB\n' if str(p)=='/proc/meminfo' else original(p,*a,**k))
    monkeypatch.setattr(target.subprocess,'run',lambda *a,**k:pytest.fail('No Docker allocation below full host reserve'))
    with pytest.raises(RuntimeError,match='complete worker allowance'):target.create_all_chip_worker(tmp_path,'software-test',memory_gib=200)
