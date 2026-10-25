import json
import pytest
from compute_trainium import future_span_execution as module
from compute_trainium.training import file_sha256


def test_resume_retains_original_budget_and_exact_file_list_ignores_unrelated_helpers(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'managed_running',lambda:[])
    now=[1000.];monkeypatch.setattr(module.time,'time',lambda:now[0])
    source=tmp_path/'frozen.py';source.write_text('x=1\n')
    plan_file=tmp_path/'plan.json'
    plan_file.write_text(json.dumps({'source_sha256':{str(source):file_sha256(source)},'seconds_limit':100,
        'phase_seconds_limit':10,'cpu_phase_seconds_limit':10,'root_comm_ports':{}}))
    out=tmp_path/'execution';first=module.Execution(tmp_path,plan_file,out)
    assert first.deadline==1100
    (tmp_path/'future_unrelated.py').write_text('x=2\n');first.check()
    first.close(tmp_path/'cost1.json');now[0]=1099
    resumed=module.Execution(tmp_path,plan_file,out,resume=True)
    assert resumed.deadline==1100
    now[0]=1101
    with pytest.raises(TimeoutError):resumed.check()
    resumed.close(tmp_path/'cost2.json')


def test_partial_native_phase_never_restarts_silently(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'managed_running',lambda:[])
    called=[];monkeypatch.setattr(module,'create_compact_worker',lambda *args:called.append(args))
    plan_file=tmp_path/'plan.json';plan_file.write_text(json.dumps({'source_sha256':{},'seconds_limit':100,
        'phase_seconds_limit':10,'cpu_phase_seconds_limit':10,'root_comm_ports':{}}))
    execution=module.Execution(tmp_path,plan_file,tmp_path/'execution')
    (tmp_path/'runs/partial').mkdir(parents=True)
    with pytest.raises(RuntimeError,match='explicit recovery'):
        execution.native('partial','scripts.not_called',[],'OK')
    assert not called
    execution.close(tmp_path/'cost.json')
