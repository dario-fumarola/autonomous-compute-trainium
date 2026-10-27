"""One conditional cell, with no idle gap and no numerical gate bypass."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts import run_scale_boundary_rate_completion as cell

IDENTITY='scale-boundary-rate-completion-queue-v1'


def prepare(root):
    folder=root/IDENTITY
    if folder.exists():raise FileExistsError('Preserve every conditional admission')
    tune=root/'scale-boundary-scope-freeze-v1/scope.json';deadline=json.loads(tune.read_text())['rate_completion_deadline_unix']
    if time.time()+9000>=deadline:raise TimeoutError('No complete remaining stage allowance')
    folder.mkdir();write_state(folder/'plan.json',dict(identity=IDENTITY,registered_unix=time.time(),
        absolute_deadline_unix=deadline,latest_admission_unix=deadline-9000,
        prerequisite_pid=4073192,prerequisite='scale-boundary-learning-lr1-v1/report.json',
        prerequisite_status='COMPLETE_FRESH_BOUNDARY_CORRECTED_LEARNING_CELL',
        source_sha256={f:file_sha256(f) for f in ['scripts/queue_scale_boundary_rate_completion.py','scripts/run_scale_boundary_rate_completion.py']},
        scope_freeze_sha256=file_sha256(tune),maximum_new_learning_updates=1505,
        scope='Wait for every mature numerical/replay/readout/closure gate of the fresh corrected0.016 cell,then admit only the five remaining fixed opportunities under the separately frozen14:45UTC completion cap. No retry,changed rate,main fit or initial24-hour deadline renewal.'))


def run(root):
    folder=root/IDENTITY;plan=json.loads((folder/'plan.json').read_text());result=dict(status='STOPPED_CONDITIONAL_BOUNDARY_RATE_STAGE',scientific_main_fit_admitted=False);child=None
    def stop(number,frame):raise KeyboardInterrupt('Conditional cell queue stopped')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        while not (root/plan['prerequisite']).exists():
            if time.time()>plan['latest_admission_unix']:raise TimeoutError('Complete measured150-minute stage no longer fits its frozen cutoff')
            if subprocess.run(['kill','-0',str(plan['prerequisite_pid'])],capture_output=True).returncode:raise RuntimeError('Corrected learning cell stopped without complete receipt')
            time.sleep(2)
        proof=json.loads((root/plan['prerequisite']).read_text())
        if proof['status']!=plan['prerequisite_status']:raise RuntimeError('Corrected learning cell failed; no automatic stage fallback')
        if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Conditional source changed')
        cell.prepare(root)
        env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONPATH':str(Path.cwd())}
        with (folder/'cell.log').open('x') as log:
            child=subprocess.Popen(['/home/ubuntu/ema-trainium/.venv/bin/python','-B','-m','scripts.run_scale_boundary_rate_completion','--root',str(root),'--run'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_state(folder/'admitted.json',dict(pid=child.pid,admitted_unix=time.time(),completed_cell_sha256=file_sha256(root/plan['prerequisite'])))
        child.wait(timeout=max(1,plan['absolute_deadline_unix']-time.time()))
        if child.returncode:raise RuntimeError('Rate-completion stage stopped;preserve receipt and no automatic retry')
        result.update(status='COMPLETE_CONDITIONAL_BOUNDARY_RATE_STAGE',cell_report_sha256=file_sha256(root/cell.IDENTITY/'report.json'))
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
