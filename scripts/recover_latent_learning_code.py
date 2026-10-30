"""Separate same-deadline rerun after a source-bundle isolation stop."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import readout_latent_learning_code as engine
from scripts.readout_direct_arithmetic import guard

OLD=engine.ID
ID=OLD+'-recovery-v1'
PUBLIC=Path('results/latent_exposure_20261030/code-readout-recovery-v1')


def prepare(root):
    p=engine.read(root/OLD/'plan.json');failure=engine.read(root/OLD/'report.json')
    if failure['status']!='STOPPED_LEARNING_CODE_READOUT' or failure['error_type']!='KeyboardInterrupt':
        raise AssertionError('Explicit preserved source-isolation stop required')
    if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Restore original frozen source before recovery')
    if p['absolute_deadline_unix']-time.time()<1000:raise RuntimeError('Measured complete readout must fit original deadline')
    p=copy.deepcopy(p);p.update(identity=ID,recovery_registered_unix=time.time(),
        recovery_of=dict(plan_sha256=file_sha256(root/OLD/'plan.json'),report_sha256=file_sha256(root/OLD/'report.json')))
    for name in ('scripts/recover_latent_learning_code.py','docs/LATENT_CODE_SOURCE_RECOVERY_20261030.md'):
        p['source_sha256'][name]=file_sha256(name)
    p['recovery_scope']=Path('docs/LATENT_CODE_SOURCE_RECOVERY_20261030.md').read_text()
    if (root/ID).exists() or PUBLIC.exists():raise FileExistsError('Preserve every recovery attempt')
    guard(root,p);(root/ID).mkdir();PUBLIC.mkdir(parents=True);write_state(root/ID/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def run(root):
    from scripts.readout_interleaved_structural import statistics
    p=engine.read(root/ID/'plan.json');start=time.time();jobs=[];logs=[];result=dict(status='STOPPED_LEARNING_CODE_RECOVERY')
    try:
        for role in engine.ROLES:
            log=(root/ID/(role+'.log')).open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.recover_latent_learning_code',
                '--root',str(root),'--action','worker','--role',role],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            guard(root,p)
            if any(j.poll() not in (None,0) for j in jobs):raise RuntimeError('Frozen recovery worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs):raise RuntimeError('Frozen recovery worker failed')
        workers={r:engine.read(root/ID/(r+'-report.json')) for r in engine.ROLES}
        if any(v['status']!='COMPLETE_LEARNING_CODE_WORKER' or len(v['checks'])!=3 or v['rows']!=564 for v in workers.values()):
            raise AssertionError('Complete cases and numerical checks required')
        if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Frozen source bundle changed')
        rows=[v for r in engine.ROLES for v in engine.read(root/ID/('PRIVATE-'+r+'-scores.json'))]
        result.update(status='COMPLETE_LEARNING_CODE_RECOVERY',typed=engine.typed_statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed']),
            raw=statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed']),workers=workers)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        for j in jobs:
            if j.poll() is None:j.terminate()
        for j in jobs:
            try:j.wait(timeout=10)
            except subprocess.TimeoutExpired:j.kill();j.wait()
        for log in logs:log.close()
        result.update(wall_seconds=time.time()-start,assigned_chips=0,optimizer_updates=0,
            protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False,
            original_deadline_unix=p['absolute_deadline_unix'],recovery_of=p['recovery_of'],
            concurrent_native_owner=p['concurrent_native_owner'],plan_sha256=file_sha256(root/ID/'plan.json'))
        write_state(root/ID/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=('prepare','run','worker'),required=True)
    p.add_argument('--role',choices=engine.ROLES);a=p.parse_args()
    if a.action=='worker':engine.ID=ID;engine.PUBLIC=PUBLIC;engine.worker(a.root,a.role)
    else:(prepare if a.action=='prepare' else run)(a.root)
