import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

from compute_trainium.training import file_sha256
from scripts import queue_scale_body_exposure_main as module


def setup(tmp_path,monkeypatch,*,stopped=False,archive_error=False):
    tune=tmp_path/module.TUNE;tune.mkdir();(tune/'plan.json').write_text('{}')
    (tune/'controller.pid').write_text(str(os.getpid()))
    (tune/'report.json').write_text(json.dumps(dict(status='STOPPED' if stopped else 'COMPLETE_BOUNDARY_LOWER_INDEPENDENT_RATE_SELECTION',plan_sha256=file_sha256(tune/'plan.json'))))
    scope=tmp_path/'scale-body-exposure-scope-v1';scope.mkdir()
    (scope/'scope.json').write_text(json.dumps(dict(recurrent_updates=15286,ordinary_work_updates=20703,
        recurrent_targets=8000125097,ordinary_work_targets=10835106429)))
    clock=tmp_path/'research-clock-accounting-amendment-20261027-v1';clock.mkdir();(clock/'plan.json').write_text('{}')
    archive=tmp_path/'closed-legacy-checkpoint-exact-archive-v3';archive.mkdir();(archive/'plan.json').write_text('{}')
    (archive/'report.json').write_text(json.dumps(dict(status='STOPPED_CLOSED_LEGACY_EXACT_ARCHIVE' if archive_error else 'COMPLETE_CLOSED_LEGACY_EXACT_ARCHIVE',error='IO failed',all_completed_original_sha256_restorable=True)))
    monkeypatch.setattr(module,'managed_running',lambda:[])
    from scripts import run_scale_boundary_lower_rates
    monkeypatch.setattr(run_scale_boundary_lower_rates,'selection_source',lambda root:(tune/'plan.json',tune/'report.json'))
    return tune


@pytest.mark.parametrize('stopped,archive_error',[(True,False),(False,True)])
def test_handoff_never_admits_main_after_failed_prerequisite(tmp_path,monkeypatch,stopped,archive_error):
    setup(tmp_path,monkeypatch,stopped=stopped,archive_error=archive_error);calls=[]
    monkeypatch.setattr(module,'prepare_storage',lambda *a,**k:calls.append('storage'))
    monkeypatch.setattr(module.engine,'prepare',lambda *a,**k:calls.append('main'))
    with pytest.raises(ValueError):module.run(tmp_path)
    assert calls==[] and not (tmp_path/module.MAIN).exists()
    result=json.loads((tmp_path/module.IDENTITY/'report.json').read_text())
    assert not result['scientific_main_fit_admitted']


@pytest.mark.parametrize('short_space',[False,True])
def test_handoff_orders_real_admission_before_main_and_keeps_fixed_scope(tmp_path,monkeypatch,short_space):
    setup(tmp_path,monkeypatch);calls=[];free=[40*1024**3 if short_space else 44*1024**3]
    monkeypatch.setattr(module,'main_state_allowance',lambda root:dict(complete_additional_bytes=8*1024**3))
    monkeypatch.setattr(module.shutil,'disk_usage',lambda path:SimpleNamespace(free=free[0]))
    def storage(root,**kwargs):
        assert kwargs=={'body_exposure':True} and free[0]>=43*1024**3
        calls.append('storage')
    def prepare(root,**kwargs):
        assert kwargs=={'boundary_corrected':True,'body_exposure':True};calls.append('prepare-main')
        folder=root/module.MAIN;folder.mkdir();(folder/'plan.json').write_text('{}')
    def run(root):
        calls.append('run-main');(root/module.MAIN/'report.json').write_text('{}')
    monkeypatch.setattr(module,'prepare_storage',storage)
    monkeypatch.setattr(module.engine,'prepare',prepare);monkeypatch.setattr(module.engine,'run',run)
    class Archive:
        returncode=0;pid=1
        def __init__(self,command,**kwargs):
            assert 'scripts.archive_closed_legacy_after_tune' in command
            calls.append('archive');free[0]=44*1024**3
            folder=tmp_path/'closed-legacy-post-tune-exact-archive-v1';folder.mkdir();(folder/'report.json').write_text('{}')
        def poll(self):return 0
    monkeypatch.setattr(module.subprocess,'Popen',Archive)
    module.run(tmp_path)
    assert calls==(['archive'] if short_space else [])+['storage','prepare-main','run-main']
    report=json.loads((tmp_path/module.IDENTITY/'report.json').read_text())
    assert report['status']=='COMPLETE_GUARDED_MAIN_HANDOFF' and report['scientific_main_fit_admitted']


def test_handoff_accepts_only_verified_archive_boundary_closure():
    good=dict(status='STOPPED_CLOSED_LEGACY_EXACT_ARCHIVE',
        error='Declared active stage is terminal;stop maintenance before any unknown successor',
        all_completed_original_sha256_restorable=True)
    module.validate_archive_closure(good)
    with pytest.raises(ValueError):module.validate_archive_closure({**good,'error':'corrupt original'})
    with pytest.raises(ValueError):module.validate_archive_closure({**good,'all_completed_original_sha256_restorable':False})
