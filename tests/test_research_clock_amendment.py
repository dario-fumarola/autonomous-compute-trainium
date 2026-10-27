from types import SimpleNamespace
import pytest
from scripts import run_scale_sharding_fixture as module


def setup_execution(monkeypatch,tmp_path,waived):
    calls=[]
    monkeypatch.setattr(module.Execution,'check',lambda self:calls.append('base-gates'))
    monkeypatch.setattr(module,'require_inode_reserve',lambda paths:calls.append('inode-gate'))
    monkeypatch.setattr(module,'SCALE_END',1)
    monkeypatch.setattr(module.time,'time',lambda:10)
    execution=module.AllChipExecution.__new__(module.AllChipExecution)
    execution.root=tmp_path
    execution.plan=dict(clock_admission_waiver=waived,absolute_deadline_unix=100)
    return execution,calls


def test_owner_clock_amendment_preserves_default_and_parent_gates(monkeypatch,tmp_path):
    execution,calls=setup_execution(monkeypatch,tmp_path,False)
    with pytest.raises(TimeoutError,match='24-hour'):execution.check()
    execution.plan['clock_admission_waiver']=True
    execution.check()
    assert calls==['base-gates','inode-gate','base-gates','inode-gate']
    def failed_parent(self):raise RuntimeError('preserved parent numerical/resource stop')
    monkeypatch.setattr(module.Execution,'check',failed_parent)
    with pytest.raises(RuntimeError,match='preserved parent'):execution.check()


def test_clock_amendment_keeps_watchdog_and_artifact_reserve(monkeypatch,tmp_path):
    execution,_=setup_execution(monkeypatch,tmp_path,True)
    execution.plan['absolute_deadline_unix']=9
    with pytest.raises(TimeoutError,match='original numerical fixture cap'):execution.check()
    execution.plan.update(absolute_deadline_unix=100,host_artifact_root=str(tmp_path))
    monkeypatch.setattr(module.shutil,'disk_usage',lambda path:SimpleNamespace(free=34*1024**3))
    with pytest.raises(RuntimeError,match='filesystem protected reserve'):execution.check()
