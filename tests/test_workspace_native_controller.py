import json
import pytest
from scripts import run_workspace_native_fixture as module


def test_failed_warm_fixture_never_starts_replay_and_preserves_stop(tmp_path,monkeypatch):
    root=tmp_path/'root';out=root/module.IDENTITY;out.mkdir(parents=True)
    public=tmp_path/'public';public.mkdir();calls=[]
    class Execution:
        sha='fixture-plan'
        plan={'root_comm_ports':{module.IDENTITY+'-warm':12345,module.IDENTITY+'-replay':12346}}
        def __init__(self,*args):pass
        def native(self,name,*args):calls.append(name);raise RuntimeError('fixture numerical failure')
        def close(self,path):calls.append('closed');path.write_text(json.dumps({'actual_worker_costs':[]}))
    monkeypatch.setattr(module,'AllChipExecution',Execution);monkeypatch.setattr(module,'PUBLIC',public)
    with pytest.raises(RuntimeError):module.run(root)
    assert calls==[module.IDENTITY+'-warm','closed']
    report=json.loads((out/'report.json').read_text())
    assert report['status']=='STOPPED_WORKSPACE_NATIVE_FIXTURE' and not report['policy_fit_admitted']
