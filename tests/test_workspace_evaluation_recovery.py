import json
import pytest
from scripts import resume_workspace_discovery_evaluation as m


def test_evaluation_recovery_requires_all_mature_audits_and_completed_budgets(tmp_path):
    plan={'updates':{'a':3},'arm_runs':{'a':'kept'}}
    native=tmp_path/'runs/kept';native.mkdir(parents=True)
    audit=tmp_path/'workspace-discovery-audit-a-v1';audit.mkdir()
    (native/'report.json').write_text(json.dumps({'status':'COMPLETE_WORKSPACE_DISCOVERY_ARM_PENDING_MATURE_AUDIT','actual_optimizer_updates':3}))
    (audit/'report.json').write_text('{"status":"FAILED"}')
    with pytest.raises(AssertionError):m.validate_endpoints(tmp_path,plan)
    (audit/'report.json').write_text('{"status":"PASS_WORKSPACE_DISCOVERY_MATURE_CPU_AUDIT"}')
    assert 'a' in m.validate_endpoints(tmp_path,plan)


def test_evaluation_failure_cannot_relaunch_training_or_score_analysis(tmp_path,monkeypatch):
    out=tmp_path/m.IDENTITY;out.mkdir();public=tmp_path/'public';public.mkdir();calls=[]
    class E:
        sha='frozen';plan={'retained_training_costs':{},'retained_endpoint_proofs':{},'root_comm_ports':{m.IDENTITY+'-evaluation':23456}}
        def __init__(self,*args):pass
        def native(self,phase,module,*args):calls.append(module);raise RuntimeError('scoring failed')
        def close(self,path):path.write_text('{}')
    monkeypatch.setattr(m,'AllChipExecution',E);monkeypatch.setattr(m,'PUBLIC',public);monkeypatch.setattr(m,'validate_endpoints',lambda *args:{})
    monkeypatch.setattr(m,'analyze',lambda *args:pytest.fail('No analysis after failed scoring'))
    with pytest.raises(RuntimeError):m.run(tmp_path)
    assert calls==['scripts.evaluate_workspace_discovery']
    assert json.loads((out/'report.json').read_text())['scientific_optimizer_updates']==0
