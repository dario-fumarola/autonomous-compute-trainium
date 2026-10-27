"""Guarded fixed-budget handoff; no retries, extra rates or confirmation."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.prepare_scale_main_storage import main_state_allowance
from scripts.prepare_scale_boundary_main_storage import prepare as prepare_storage
from scripts import run_scale_main as engine

IDENTITY='scale-body-exposure-main-handoff-v1'
TUNE='scale-boundary-lower-rates-v1'
MAIN='scale-body-exposure-main-v1'


def validate_completed_tune(report,plan_sha):
    if (report['status']!='COMPLETE_BOUNDARY_LOWER_INDEPENDENT_RATE_SELECTION'
            or report['plan_sha256']!=plan_sha):
        raise ValueError('Tuning stopped or differs from its frozen registration;no automatic recovery')


def validate_archive_closure(report):
    complete=report['status']=='COMPLETE_CLOSED_LEGACY_EXACT_ARCHIVE'
    boundary=(report['status']=='STOPPED_CLOSED_LEGACY_EXACT_ARCHIVE'
              and report.get('error')=='Declared active stage is terminal;stop maintenance before any unknown successor')
    if not (complete or boundary) or not report.get('all_completed_original_sha256_restorable'):
        raise ValueError('Archival failed beyond its declared boundary or lost verification;inspect before main')


def run(root):
    folder=root/IDENTITY
    if folder.exists():raise FileExistsError('Preserve every handoff attempt')
    if (root/MAIN).exists():raise FileExistsError('No second main admission')
    tune=root/TUNE;scope=root/'scale-body-exposure-scope-v1/scope.json'
    archive=root/'closed-legacy-checkpoint-exact-archive-v3'
    clock=root/'research-clock-accounting-amendment-20261027-v1/plan.json'
    specification=json.loads(scope.read_text())
    if tuple(specification[k] for k in ['recurrent_updates','ordinary_work_updates','recurrent_targets','ordinary_work_targets'])!=(15286,20703,8000125097,10835106429):
        raise AssertionError('Exact registered optimizer and source endpoints required')
    if file_sha256(Path(__file__))!=file_sha256('scripts/queue_scale_body_exposure_main.py'):
        raise AssertionError('Run only from the frozen execution checkout')
    sources=[Path('scripts/queue_scale_body_exposure_main.py'),Path('scripts/run_scale_main.py'),Path('scripts/prepare_scale_boundary_main_storage.py'),
             Path('scripts/archive_closed_legacy_after_tune.py'),Path('scripts/run_scale_body_exposure_main.py'),
             Path('docs/SCALE_BODY_EXPOSURE_HANDOFF_20261027.md')]
    plan=dict(identity=IDENTITY,registered_unix=time.time(),operational_wait_seconds=8*3600,
        tuning_plan_sha256=file_sha256(tune/'plan.json'),tuning_pid=int((tune/'controller.pid').read_text()),
        scientific_scope_sha256=file_sha256(scope),clock_amendment_sha256=file_sha256(clock),
        preceding_archival_plan_sha256=file_sha256(archive/'plan.json'),
        source_sha256={str(p):file_sha256(p) for p in sources},
        inactive_archive_maximum_original_bytes=50*1024**3,inactive_archive_maximum_attempts=1,
        main_identity=MAIN,main_optimizer_endpoints=dict(recurrent=15286,ordinary_work=20703),
        scientific_main_fit_admitted=False,
        scope='Wait for exact complete independent TUNE and archival closure. On failure stop,never retry rates or weaken gates. If needed perform one at-most50GiB inactive exact archive of closed historical binaries. Only fresh full storage,numerical,source,pristine-C0 and scope admission may start the fixed8B/10.835B main. All old failures retained. No automatic confirmation or further science.')
    folder.mkdir();write_state(folder/'plan.json',plan)
    result=dict(status='STOPPED_GUARDED_MAIN_HANDOFF',scientific_main_fit_admitted=False)
    child=None;original_engine_identity=engine.IDENTITY;original_engine_public=engine.PUBLIC
    def stop(number,frame):raise KeyboardInterrupt('Guarded main handoff stopped')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    def bindings():
        if (file_sha256(tune/'plan.json')!=plan['tuning_plan_sha256']
                or file_sha256(scope)!=plan['scientific_scope_sha256']
                or file_sha256(clock)!=plan['clock_amendment_sha256']
                or file_sha256(archive/'plan.json')!=plan['preceding_archival_plan_sha256']
                or any(file_sha256(p)!=sha for p,sha in plan['source_sha256'].items())):
            raise AssertionError('Frozen handoff source or prerequisite changed')
    try:
        while True:
            bindings()
            if time.time()>plan['registered_unix']+plan['operational_wait_seconds']:
                raise TimeoutError('Bounded operational wait ended;no automatic renewal')
            if (tune/'report.json').exists():
                validate_completed_tune(json.loads((tune/'report.json').read_text()),plan['tuning_plan_sha256'])
                if (archive/'report.json').exists():
                    validate_archive_closure(json.loads((archive/'report.json').read_text()))
                    if not managed_running():break
            elif subprocess.run(['kill','-0',str(plan['tuning_pid'])],capture_output=True).returncode:
                raise RuntimeError('Tuning handle missing without a terminal receipt')
            write_state(folder/'progress.json',dict(status='WAITING_REGISTERED_MAIN_GATES',scientific_main_fit_admitted=False,
                elapsed_seconds=time.time()-plan['registered_unix']))
            time.sleep(10)
        from scripts.run_scale_boundary_lower_rates import selection_source
        _,selected_report=selection_source(root)
        allowance=main_state_allowance(root);required=35*1024**3+allowance['complete_additional_bytes']
        if shutil.disk_usage(root).free<required:
            log=folder/'inactive-archive.log'
            env=dict(os.environ,USE_TORCH_XLA='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path('.').resolve()))
            command=['sudo','-n','env','USE_TORCH_XLA=0','PYTHONDONTWRITEBYTECODE=1',
                'PYTHONPATH='+str(Path('.').resolve()),os.sys.executable,'-B','-m','scripts.archive_closed_legacy_after_tune',
                '--root',str(root),'--required-free-bytes',str(required),'--tuning-report',str(selected_report),
                '--scope-file',str(scope)]
            with log.open('x') as output:
                child=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                started=time.time()
                while child.poll() is None:
                    bindings()
                    if time.time()>started+5500:raise TimeoutError('Bounded inactive preservation watchdog')
                    time.sleep(5)
                if child.returncode:raise RuntimeError('Bounded inactive preservation failed;no automatic extension')
            result['inactive_archive_report_sha256']=file_sha256(root/'closed-legacy-post-tune-exact-archive-v1/report.json')
            child=None
        bindings()
        prepare_storage(root,body_exposure=True)
        engine.IDENTITY=MAIN;engine.PUBLIC=Path('results/scale_body_exposure_main_20261027')
        engine.prepare(root,boundary_corrected=True,body_exposure=True)
        result['scientific_main_fit_admitted']=True
        result['main_plan_sha256']=file_sha256(root/MAIN/'plan.json')
        write_state(folder/'progress.json',{**result,'status':'RUNNING_REGISTERED_MAIN'})
        engine.run(root)
        result.update(status='COMPLETE_GUARDED_MAIN_HANDOFF',main_report_sha256=file_sha256(root/MAIN/'report.json'))
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        if child and child.poll() is None:
            subprocess.run(['sudo','-n','kill','-TERM','--','-'+str(child.pid)],capture_output=True)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                subprocess.run(['sudo','-n','kill','-KILL','--','-'+str(child.pid)],capture_output=True);child.wait(timeout=10)
        result['wall_seconds']=time.time()-plan['registered_unix']
        result['cost_scope']='Handoff wall overlaps TUNE wait and the complete main study;do not add it again to their opportunity or assigned-chip ledgers.'
        write_state(folder/'report.json',result)
        engine.IDENTITY=original_engine_identity;engine.PUBLIC=original_engine_public
        for s,handler in handlers.items():signal.signal(s,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
