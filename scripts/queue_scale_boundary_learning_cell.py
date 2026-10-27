"""One conditional cell, with no idle gap and no numerical gate bypass."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts import run_scale_boundary_learning_cell as cell

IDENTITY='scale-boundary-learning-cell-queue-v1'


def prepare(root):
    folder=root/IDENTITY
    if folder.exists():raise FileExistsError('Preserve every conditional admission')
    tune=root/'scale-medium-tune-65536-v1/plan.json';deadline=json.loads(tune.read_text())['absolute_deadline_unix']
    if time.time()+2100>=deadline:raise TimeoutError('No remaining complete cell allowance')
    folder.mkdir();write_state(folder/'plan.json',dict(identity=IDENTITY,registered_unix=time.time(),
        absolute_deadline_unix=deadline,latest_admission_unix=deadline-2100,
        prerequisite_pid=4023589,prerequisite='scale-boundary-fixed-two-numerical-v1/completion-verification.json',
        prerequisite_status='PASS_BOUNDARY_COMPLETE_NUMERICAL_QUALIFICATION',
        source_sha256={f:file_sha256(f) for f in ['scripts/queue_scale_boundary_learning_cell.py','scripts/run_scale_boundary_learning_cell.py']},
        original_tune_plan_sha256=file_sha256(tune),maximum_new_learning_updates=301,
        scope='Wait only for complete fresh numerical qualification,then admit the single corrected0.016 cell if every source,resource and full35-minute original-clock gate passes. No retry,rate selection,main fit or deadline renewal.'))


def run(root):
    folder=root/IDENTITY;plan=json.loads((folder/'plan.json').read_text());result=dict(status='STOPPED_CONDITIONAL_BOUNDARY_CELL',scientific_main_fit_admitted=False);child=None
    def stop(number,frame):raise KeyboardInterrupt('Conditional cell queue stopped')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        while not (root/plan['prerequisite']).exists():
            if time.time()>plan['latest_admission_unix']:raise TimeoutError('Complete35-minute cell no longer fits original TUNE cutoff')
            if subprocess.run(['kill','-0',str(plan['prerequisite_pid'])],capture_output=True).returncode:raise RuntimeError('Qualification stopped without complete receipt')
            time.sleep(2)
        proof=json.loads((root/plan['prerequisite']).read_text())
        if proof['status']!=plan['prerequisite_status']:raise RuntimeError('Qualification failed; no automatic learning fallback')
        if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Conditional source changed')
        cell.prepare(root)
        env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONPATH':str(Path.cwd())}
        with (folder/'cell.log').open('x') as log:
            child=subprocess.Popen(['/home/ubuntu/ema-trainium/.venv/bin/python','-B','-m','scripts.run_scale_boundary_learning_cell','--root',str(root),'--run'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_state(folder/'admitted.json',dict(pid=child.pid,admitted_unix=time.time(),qualification_sha256=file_sha256(root/plan['prerequisite'])))
        child.wait(timeout=max(1,plan['absolute_deadline_unix']-time.time()))
        if child.returncode:raise RuntimeError('Single learning cell stopped; preserve receipt and no automatic retry')
        result.update(status='COMPLETE_SINGLE_CONDITIONAL_BOUNDARY_CELL',cell_report_sha256=file_sha256(root/cell.IDENTITY/'report.json'))
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        result['finished_unix']=time.time();write_state(folder/'report.json',result)
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
