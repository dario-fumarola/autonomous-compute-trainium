import json
import pytest
from scripts import run_workspace_online_throughput as module


def test_partial_online_failure_preserves_observed_updates_and_never_admits_fit(tmp_path,monkeypatch):
    root=tmp_path/'root';out=root/module.IDENTITY;out.mkdir(parents=True)
    public=tmp_path/'public';public.mkdir()
    phase=module.IDENTITY+'-online';calls=[]
    class Execution:
        sha='fixed-plan';plan={'root_comm_ports':{phase:12345}}
        def __init__(self,*args):pass
        def native(self,*args):
            run=root/'runs'/phase;run.mkdir(parents=True)
            (run/'updates.json').write_text(json.dumps([{'update':1}]))
            raise RuntimeError('owned numerical stop')
        def close(self,path):calls.append('closed');path.write_text(json.dumps({'native_attempts':[]}))
    monkeypatch.setattr(module,'AllChipExecution',Execution);monkeypatch.setattr(module,'PUBLIC',public)
    with pytest.raises(RuntimeError):module.run(root)
    report=json.loads((out/'report.json').read_text())
    assert report['observed_completed_qualification_optimizer_updates']==1
    assert report['interrupted_update_may_be_unrecorded'] and not report['policy_fit_admitted']
    assert calls==['closed']
