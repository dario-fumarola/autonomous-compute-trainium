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


def test_recovery_reuses_audited_ordinary_and_stops_on_workspace_failure(tmp_path,monkeypatch):
    from compute_trainium.training import file_sha256
    root=tmp_path/'root';out=root/module.IDENTITY;out.mkdir(parents=True);public=tmp_path/'public';public.mkdir()
    native=root/'ordinary.json';audit=root/'audit.json';native.write_text('{"actual_optimizer_updates":931}');audit.write_text('{"status":"PASS"}')
    phase=module.IDENTITY+'-workspace';calls=[]
    reused_plan={'root_comm_ports':{phase:23456},'reused_ordinary':{'native_report_relative':'ordinary.json','native_report_sha256':file_sha256(native),'audit_relative':'audit.json','audit_sha256':file_sha256(audit)},'prior_execution_costs':{'retained':True}}
    class Execution:
        sha='fixed';plan=reused_plan
        def __init__(self,*args):pass
        def native(self,identity,*args):calls.append(identity);raise RuntimeError('owned workspace stop')
        def close(self,path):path.write_text('{}')
    monkeypatch.setattr(module,'AllChipExecution',Execution);monkeypatch.setattr(module,'PUBLIC',public)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda p:type('Usage',(),{'free':1000*1024**3})())
    with pytest.raises(RuntimeError):module.run(root)
    a=json.loads((out/'report.json').read_text())
    assert calls==[phase] and a['conditions']['ordinary']['reused_completed_endpoint']
    assert a['prior_execution_costs']=={'retained':True}
