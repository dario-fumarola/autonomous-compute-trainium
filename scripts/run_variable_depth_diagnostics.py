"""Time-bounded read-only annex; never competes with the main native campaign."""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import queue
import subprocess
import time
from pathlib import Path
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);deadline=dt.datetime(2026,10,24,4,15,tzinfo=dt.timezone.utc).timestamp()
    started=time.time();cpu=None;public=Path('results/variable_depth_20261023');croot='/work/instance-nvme'
    spec={'annex_sha256':file_sha256('docs/VARIABLE_DEPTH_READONLY_DIAGNOSTICS_20261023.md'),
          'source_sha256':{f:file_sha256(f) for f in ['scripts/audit_depth_mechanism.py','scripts/run_variable_depth_diagnostics.py']},
          'declared_unix':started,'deadline_unix':deadline}
    write_state(a.output/'specification.json',spec);write_state(public/'readonly-diagnostics-specification.json',spec)
    while not (a.campaign/'test-release.json').exists():
        if (a.campaign/'report.json').exists() or time.time()>deadline:
            write_state(a.output/'report.json',{'status':'UNEXECUTED_PRIMARY_PREREQUISITE'});return
        time.sleep(10)
    if time.time()+3000<deadline:
        env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12')
        log=(a.output/'independent-mechanism.log').open('w')
        cpu=subprocess.Popen([os.sys.executable,'-u','-m','scripts.audit_depth_mechanism','--root',str(a.root),
             '--corpus',str(a.root/'variable-depth-fresh-v1/dev.json'),'--output',str(a.output/'independent-mechanism'),
             '--deadline-unix',str(deadline-30)],env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close();write_state(a.output/'cpu-process.json',{'pid':cpu.pid,'started_unix':time.time()})
    while not (a.campaign/'report.json').exists():
        if time.time()>deadline:raise TimeoutError('Primary campaign waiting deadline')
        time.sleep(10)
    result=json.loads((a.campaign/'report.json').read_text())
    if result['status']!='COMPLETE_VARIABLE_DEPTH_CAMPAIGN':
        if cpu and cpu.poll() is None:cpu.terminate()
        write_state(a.output/'report.json',{'status':'UNEXECUTED_PRIMARY_FAILED'});return
    rows=[]
    if time.time()+1500<deadline:
        tasks=queue.Queue()
        for arm in ['fixed','mixture']:
            for seed in [103,211,307]:
                for update in [384,1152]:
                    for depth in [1,3]:tasks.put((arm,seed,update,depth))
        plan=json.loads((a.campaign/'plan.json').read_text())
        def worker(half):
            records=[]
            while not tasks.empty():
                try:arm,seed,update,depth=tasks.get_nowait()
                except queue.Empty:break
                if time.time()+120>deadline:raise TimeoutError('Annex readout deadline')
                for f,sha in plan['source_sha256'].items():
                    if file_sha256(f)!=sha:raise AssertionError('Main frozen sources changed')
                name=f'vdepth-diag-v1-{arm}-s{seed}-u{update}-dev-r{depth}'
                env=dict(os.environ,RESEARCH_ROOT=str(a.root),RESEARCH_CONTAINER_ROOT=croot,RESEARCH_RUN=name,
                    RESEARCH_ENTRY_MODULE='scripts.evaluate_depth_curve',RESEARCH_DDP_RANKS='8',
                    NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,(half+1)*32,4))),
                    RESEARCH_TIMEOUT_SECONDS=str(min(600,int(deadline-time.time()))))
                args=['--corpus',f'{croot}/variable-depth-fresh-v1/dev.json','--checkpoint',
                    f'{croot}/runs/vdepth-v1-{arm}-s{seed}-u{update}/checkpoint.pt',
                    '--repetitions',str(depth),'--cohort','curve','--split','dev']
                log=(a.output/(name+'.log')).open('w')
                result=subprocess.run([os.sys.executable,'-u','-m','scripts.supervise',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
                log.close()
                if result.returncode:raise RuntimeError('Annex native readout failed: '+name)
                report=json.loads((a.root/'runs'/name/'report.json').read_text())
                records.append({'identity':name,'arm':arm,'seed':seed,'update':update,'depth':depth,
                       'training_real_targets':report['training_real_tokens'],'metrics':report['metrics']})
                print(json.dumps(records[-1]),flush=True)
            return records
        with cf.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(worker,h) for h in [0,1]]
            rows=[r for f in futures for r in f.result()]
    if cpu:
        cpu.wait(timeout=max(1,deadline-time.time()))
        if cpu.returncode:raise RuntimeError('Independent mechanism audit failed')
        write_state(public/'independent-mechanism-verification.json',json.loads((a.output/'independent-mechanism/report.json').read_text()))
    report={'status':'COMPLETE_READONLY_DEPTH_DIAGNOSTICS' if len(rows)==24 else 'UNEXECUTED_NATIVE_TIME_GATE',
         'native_cells':rows,'cpu_audit_executed':cpu is not None,'wall_seconds_including_waits':time.time()-started,
         'scope':'Prospective DEV-only early-checkpoint readouts and independent numerical checks. No new fit, model selection or TEST use.'}
    write_state(a.output/'report.json',report);write_state(public/'readonly-diagnostics.json',report)


if __name__=='__main__':main()
