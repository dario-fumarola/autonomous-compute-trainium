"""Software file/process fixtures only; no research examples or native jobs."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.recovery_support import (ExistingAudit,archive_partial_audit,planned_stages,
    preserve_test_release,process_matches,remaining_budget,stop_verified_process,verify_saved_sources)


def plan():
    return {'seeds':[103,211,307],'lr_candidates':[.00015,.0003,.0006,.001],
            'budget':{'measured_targets_per_second_per_half_by_arm':{'ordinary':1000,'fixed':800,'mixture':600}}}


def test_saved_manifest_membership_ignores_added_helpers_but_rejects_changed_source(tmp_path):
    frozen=tmp_path/'train.py';frozen.write_text('frozen source')
    p={'source_sha256':{'train.py':hashlib.sha256(frozen.read_bytes()).hexdigest()}}
    (tmp_path/'future.py').write_text('unrelated helper')
    assert verify_saved_sources(tmp_path,p)['verified_paths']==1
    frozen.write_text('changed source')
    with pytest.raises(ValueError):verify_saved_sources(tmp_path,p)


def test_primary_stage_catalog_and_remaining_budget_reuse_only_successful_closures(tmp_path):
    count=lambda u:u*10
    stages=planned_stages(plan(),count)
    assert len(stages)==144
    assert sum(s['kind']=='train' for s in stages.values())==54
    before=remaining_budget(plan(),tmp_path,count)
    name='vdepth-v1-tune-ordinary-lr150'
    run=tmp_path/'runs'/name;run.mkdir(parents=True)
    snap=tmp_path/'snapshots'/name;snap.mkdir(parents=True)
    (run/'report.json').write_text(json.dumps({'status':'PASS_PACKED_TRAINING_PHASE'}))
    (snap/'closure.json').write_text(json.dumps({'exit_code':1,'wall_seconds':72}))
    assert remaining_budget(plan(),tmp_path,count)['complete_stages']==0
    (snap/'closure.json').write_text(json.dumps({'exit_code':0,'wall_seconds':72}))
    after=remaining_budget(plan(),tmp_path,count)
    assert after['complete_stages']==1 and name not in after['pending_stages']
    assert after['remaining_wall_seconds_upper']<before['remaining_wall_seconds_upper']


def test_existing_test_release_keeps_timestamp_and_rejects_changed_weights():
    old={'frozen_unix':10,'models':[{'identity':'a','checkpoint_sha256':'old'}],'corpus_sha256':'panel'}
    assert preserve_test_release(dict(old,frozen_unix=999),old) is old
    with pytest.raises(ValueError):preserve_test_release(dict(old,models=[{'identity':'a','checkpoint_sha256':'new'}]),old)


def test_incomplete_readback_copy_is_archived_while_verified_copy_is_preserved(tmp_path):
    (tmp_path/'recovery/good').mkdir(parents=True);(tmp_path/'recovery/partial').mkdir()
    (tmp_path/'recovery/partial/checkpoint.partial').write_bytes(b'preserved fixture')
    (tmp_path/'progress.json').write_text(json.dumps({'records':[{'identity':'good'}]}))
    assert archive_partial_audit(tmp_path)==['partial']
    assert (tmp_path/'recovery/good').is_dir()
    assert next((tmp_path/'interrupted').rglob('checkpoint.partial')).read_bytes()==b'preserved fixture'


def test_verified_cpu_handle_is_adopted_and_cleanup_cannot_kill_wrong_output(tmp_path):
    package=tmp_path/'scripts';package.mkdir();(package/'__init__.py').touch()
    (package/'audit_depth_mechanism.py').write_text('import time\ntime.sleep(60)\n')
    output=tmp_path/'audit';env=dict(os.environ,PYTHONPATH=str(tmp_path))
    child=subprocess.Popen([sys.executable,'-m','scripts.audit_depth_mechanism','--output',str(output)],cwd=tmp_path,env=env)
    try:
        deadline=time.monotonic()+5
        while not process_matches(child.pid,'scripts.audit_depth_mechanism',output) and time.monotonic()<deadline:time.sleep(.01)
        assert process_matches(child.pid,'scripts.audit_depth_mechanism',output)
        handle=ExistingAudit(child.pid,'scripts.audit_depth_mechanism',output)
        assert handle.poll() is None
        assert not stop_verified_process(child.pid,'scripts.audit_depth_mechanism',tmp_path/'other')['matching_live_child']
        assert child.poll() is None
        result=stop_verified_process(child.pid,'scripts.audit_depth_mechanism',output)
        assert result['terminal'] and child.wait(timeout=5)!=0 and handle.poll()==1
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_annex_guard_cleans_orphan_and_writes_terminal_record_end_to_end(tmp_path):
    package=tmp_path/'scripts';package.mkdir();(package/'__init__.py').touch()
    (package/'audit_depth_mechanism.py').write_text('import time\ntime.sleep(60)\n')
    annex=tmp_path/'annex';annex.mkdir()
    (annex/'specification.json').write_text(json.dumps({'deadline_unix':time.time()+300}))
    output=annex/'independent-mechanism';env=dict(os.environ,PYTHONPATH=str(tmp_path))
    child=subprocess.Popen([sys.executable,'-m','scripts.audit_depth_mechanism','--output',str(output)],cwd=tmp_path,env=env)
    try:
        deadline=time.monotonic()+5
        while not process_matches(child.pid,'scripts.audit_depth_mechanism',output) and time.monotonic()<deadline:time.sleep(.01)
        repo=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,'-m','scripts.guard_depth_diagnostics','--annex',str(annex),
            '--pid','99999999','--output',str(tmp_path/'guard'),'--public-output',str(tmp_path/'public.json')],
            cwd=repo,env=dict(os.environ,PYTHONPATH=str(repo)),capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
        assert child.wait(timeout=5)!=0
        assert json.loads((annex/'report.json').read_text())['status']=='STOPPED_DIAGNOSTIC_CONTROLLER_WITHOUT_REPORT'
        guard=json.loads((tmp_path/'public.json').read_text())
        assert guard['status']=='REPAIRED_MISSING_DIAGNOSTIC_TERMINAL_REPORT'
        assert guard['cpu_child_cleanup'][0]['terminal']
    finally:
        if child.poll() is None:child.kill();child.wait()
