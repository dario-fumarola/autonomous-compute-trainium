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


def test_exclusive_owner_has_no_half_and_uses_same_atomic_lock(tmp_path,monkeypatch):
    env=tmp_path/'isolated-native-env-v1';env.mkdir()
    (env/'parent-private.json').write_text(json.dumps(dict(HostConfig=dict(Devices=[]),Mounts=[])))
    (env/'setup-private.json').write_text(json.dumps(dict(image='software-test')))
    monkeypatch.setattr(target,'managed_running',lambda:[])
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args,1 if args[1]=='inspect' else 0)
    monkeypatch.setattr(target.subprocess,'run',run)
    monkeypatch.setattr(target,'metadata',lambda name:dict(Config=dict(Labels={'ac-trainium-managed':'1','ac-trainium-owner':'software-test'})))
    assert target.create_all_chip_worker(tmp_path,'software-test')=='ac-worker-software-test'
    assert (tmp_path/'native-half-reservations.lock').exists()
    args=calls[-1]
    assert 'ac-trainium-chips=all16' in args
    assert not any('ac-trainium-half' in v for v in args)
    assert args[args.index('--memory')+1]=='128g'
