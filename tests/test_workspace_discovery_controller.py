import json
import pytest
from scripts import run_workspace_discovery as module


def test_discovery_failure_stops_other_conditions_and_evaluation(tmp_path,monkeypatch):
    root=tmp_path/'root';out=root/module.IDENTITY;out.mkdir(parents=True)
    public=tmp_path/'public';public.mkdir();calls=[]
    phase=module.IDENTITY+'-ordinary'
    class Execution:
        sha='frozen';plan={'root_comm_ports':{phase:23456}}
        def __init__(self,*args):pass
        def native(self,*args):calls.append('ordinary');raise RuntimeError('independent gate failed')
        def cpu(self,*args):raise AssertionError('No continuation after failure')
        def close(self,path):calls.append('closed');path.write_text(json.dumps({'native_attempts':[]}))
    monkeypatch.setattr(module,'AllChipExecution',Execution);monkeypatch.setattr(module,'PUBLIC',public)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda p:type('Usage',(),{'free':1000*1024**3})())
    with pytest.raises(RuntimeError,match='independent gate'):module.run(root)
    result=json.loads((out/'report.json').read_text())
    assert result['status']=='STOPPED_WORKSPACE_DISCOVERY_PILOT' and result['conditions']=={}
    assert calls==['ordinary','closed'] and not result['full_from_scratch_admitted']
