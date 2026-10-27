import json
import queue
import threading

import pytest

import compute_trainium.future_span_execution as engine


def execution(tmp_path):
    value=object.__new__(engine.Execution)
    value.plan={'phase_seconds_limit':900,'native_phase_seconds_limits':{'fixture':300}}
    value.root=tmp_path;value.out=tmp_path/'control';value.out.mkdir()
    value.failed=threading.Event();value.lock=threading.Lock();value.cleanup_errors=[]
    value.native_active={};value.halves=queue.Queue();value.halves.put(0)
    value.deadline=10**20;value.sha='fixture-plan';value.check=lambda:None
    return value


def test_registered_phase_cap_cannot_expand_common_limit(tmp_path):
    value=execution(tmp_path)
    assert value.native_seconds_limit('fixture')==300
    assert value.native_seconds_limit('normal')==900
    for wrong in [0,-1,901,float('nan'),True]:
        value.plan['native_phase_seconds_limits']['fixture']=wrong
        with pytest.raises(ValueError):value.native_seconds_limit('fixture')


def test_cleanup_error_does_not_replace_original_worker_failure(tmp_path,monkeypatch):
    value=execution(tmp_path)
    monkeypatch.setattr(engine,'reclaim_host',lambda:None)
    monkeypatch.setattr(engine,'create_compact_worker',lambda *a:'owned-fixture')
    class Process:
        returncode=1
        def poll(self):return 1
    monkeypatch.setattr(engine.subprocess,'Popen',lambda *a,**kw:Process())
    def cleanup(*a):raise RuntimeError('secondary stop failure')
    monkeypatch.setattr(engine,'cancel_worker_container',cleanup)
    with pytest.raises(RuntimeError,match='Native phase failed: fixture'):
        value.native('fixture','unused',[],'expected')
    assert value.failed.is_set() and value.native_active=={'fixture':'owned-fixture'}
    assert value.cleanup_errors[0]['primary_error']=='Native phase failed: fixture'
    assert value.cleanup_errors[0]['error']=='secondary stop failure'


def test_failed_cleanup_still_rejects_an_otherwise_successful_phase(tmp_path,monkeypatch):
    value=execution(tmp_path);folder=tmp_path/'runs/fixture';folder.mkdir(parents=True)
    # The directory is created by the fake supervisor after admission.
    folder.rmdir()
    monkeypatch.setattr(engine,'reclaim_host',lambda:None)
    monkeypatch.setattr(engine,'create_compact_worker',lambda *a:'owned-fixture')
    class Process:
        returncode=0
        def poll(self):return 0
    def launch(*a,**kw):
        folder.mkdir();(folder/'report.json').write_text(json.dumps({'status':'expected'}));return Process()
    monkeypatch.setattr(engine.subprocess,'Popen',launch)
    def cleanup(*a):raise RuntimeError('stop failure after success')
    monkeypatch.setattr(engine,'cancel_worker_container',cleanup)
    with pytest.raises(RuntimeError,match='stop failure after success'):
        value.native('fixture','unused',[],'expected')
    assert value.failed.is_set() and value.cleanup_errors[0]['primary_error'] is None
