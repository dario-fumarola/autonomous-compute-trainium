"""Bounded continuation of the already-declared stronger-control comparisons."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_lean_baseline_main import identity

IDENTITY='remaining-lean-controls-queue-v1'
PUBLIC=Path('results/lean_baseline_controls_20261026/remaining-queue')
DOC='docs/REMAINING_LEAN_CONTROLS_QUEUE_20261026.md'
ORDER=['ordinary8','one']


def read(file):return json.loads(Path(file).read_text())


def require_complete_control(report,role):
    seeds={'5231','17293'}
    if (report['status']!='COMPLETE_QUALIFIED_LEAN_BASELINE_MAIN_CONTROL'
            or report['model_variant']!=role or set(report['scores'])!=seeds
            or set(report['final_gradient_gates'])!=seeds or set(report['recurrence_advantage'])!=seeds):
        raise ValueError('Both frozen seeds and their complete control reports are required')
    for seed in seeds:
        score=report['scores'][seed];gold=report['final_gradient_gates'][seed]['independent_cpu']
        if (score['status']!='COMPLETE_INDEPENDENT_LOW_DEPTH_DEVELOPMENT_SCORE'
                or gold['status']!='PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT'
                or score['checkpoint_sha256']!=gold['checkpoint_sha256']):
            raise ValueError('Exact endpoint score and passing mature gradient audit required')
    # Quality determines the research conclusion,never whether to hide a control.
    return True


def register(root):
    out=root/IDENTITY
    if out.exists():raise FileExistsError('Preserve queue execution;no silent restart or deadline renewal')
    active_file=root/'lean-baseline-main-mixing-v1/plan.json';active=read(active_file)
    started_file=active_file.parent/'execution/execution-start.json';started=read(started_file)
    if started['plan_sha256']!=file_sha256(active_file):raise AssertionError('Original active main identity changed')
    sources=dict(active['source_sha256'])
    for f in ['scripts/run_remaining_lean_controls.py','tests/test_remaining_lean_controls.py',DOC,
              'scripts/sparsify_closed_numerical_artifacts.py','scripts/reclaim_closed_snapshot_bytecode.py']:
        sources[f]=file_sha256(f)
    if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Frozen control/maintenance sources changed')
    deadline=started['started_unix']+12*3600
    remaining_current=max(0,active['measured_budget']['complete_seconds']-(time.time()-started['started_unix']))
    from scripts.run_lean_baseline_main import project_main_seconds
    prep=read(root/'lean-baseline-control-preparation-v2/manifest.json')
    qual=read(root/'baseline-numerical-recovery-v1/report.json')
    future={r:project_main_seconds(r,prep['profiles'][r],qual)['complete_seconds'] for r in ORDER}
    complete=remaining_current+sum(future.values())+2*600+2*180+600
    if complete>deadline-time.time():raise RuntimeError('Complete remaining controls plus maintenance do not fit original composite cap')
    plan=dict(identity=IDENTITY,registered_unix=time.time(),source_sha256=sources,
        active_main_plan=str(active_file.relative_to(root)),active_main_plan_sha256=file_sha256(active_file),
        active_main_started_sha256=file_sha256(started_file),
        original_composite_started_unix=started['started_unix'],absolute_deadline_unix=deadline,
        maximum_composite_seconds=12*3600,remaining_controls=ORDER,seeds=[5231,17293],
        measured_remaining_projection=dict(current=remaining_current,future=future,
            bounded_storage_and_inode_maintenance=1560,margin=600,total_seconds=complete),
        maintenance_only_when_admission_requires=True,publication_required_before_new_fit=True,
        per_control_seconds_limit=14400,no_retry_or_cooled_extension=True,automatic_4b_admission=False,
        test_accessed=False,terminal_alpha_spent=0,scope=Path(DOC).read_text())
    out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan


def publish(paths,message):
    subprocess.run(['git','add','--',*map(str,paths)],check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only','-z']).decode().split('\0')
    allowed=[str(p).rstrip('/') for p in paths]
    if any(f and not any(f==p or f.startswith(p+'/') for p in allowed) for f in staged):
        raise RuntimeError('Unrelated staged changes must not enter automatic aggregate publication')
    result=subprocess.run(['git','diff','--cached','--quiet'],check=False)
    if result.returncode==0:return
    if result.returncode!=1:raise RuntimeError('Cannot inspect staged aggregate publication')
    subprocess.run(['git','commit','-m',message],check=True,timeout=120)
    subprocess.run(['git','push'],check=True,timeout=120)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    out=a.root/IDENTITY;file=out/'plan.json'
    plan=read(file) if file.exists() else register(a.root)
    if a.prepare_only:return
    lock_file=(out/'controller.lock').open('a');fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'started.json').exists():raise FileExistsError('Explicit preserved recovery required for any previously started queue')
    sha=file_sha256(file);write_state(out/'started.json',dict(unix=time.time(),plan_sha256=sha))
    process=None;records={};maintenance=[];result=dict(status='STOPPED_REMAINING_LEAN_CONTROLS_QUEUE')
    def interrupted(number,frame):raise KeyboardInterrupt('Remaining control queue interrupted: '+str(number))
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def guard():
        if time.time()>plan['absolute_deadline_unix']:raise TimeoutError('Original absolute twelve-hour composite deadline')
        if file_sha256(file)!=sha or any(file_sha256(f)!=s for f,s in plan['source_sha256'].items()):raise AssertionError('Frozen queue/control source changed')
    def command(label,module,opts,limit):
        nonlocal process
        guard();log=out/(label+'.log');started=time.time()
        with log.open('x') as handle:
            process=subprocess.Popen([sys.executable,'-u','-m',module,*opts],stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            guard()
            if time.time()-started>limit:raise TimeoutError('Bounded queue phase cap: '+label)
            time.sleep(5)
        code=process.returncode;process=None
        if code:raise RuntimeError('Queue phase failed;preserve log and exact predecessor: '+label)
    def wait_control(role):
        report=a.root/identity(role)/'report.json'
        while not report.exists():
            guard()
            write_state(PUBLIC/'status.json',dict(status='WAITING_FOR_ADMITTED_CONTROL',control=role,
                no_new_run_admitted=True,absolute_deadline_unix=plan['absolute_deadline_unix']))
            time.sleep(5)
        row=read(report);require_complete_control(row,role)
        records[role]=dict(report_sha256=file_sha256(report),recurrence_advantage=row['recurrence_advantage'])
        write_state(PUBLIC/'completed-controls.json',dict(records=records))
        return row
    def prepare_resources(role):
        guard()
        if managed_running():raise RuntimeError('Previous native owners must close before next admission/maintenance')
        if shutil.disk_usage(a.root).free<47*1024**3:
            target=a.root/(IDENTITY+'-'+role+'-zero-storage')
            command(role+'-lossless-storage','scripts.sparsify_closed_numerical_artifacts',
                ['--root',str(a.root),'--output',str(target),'--apply'],660)
            row=read(target/'report.json')
            if row['status']!='COMPLETE_BYTE_EXACT_CLOSED_NUMERICAL_ZERO_STORAGE':raise ValueError('Lossless storage maintenance failed')
            maintenance.append(dict(kind='byte_exact_zero_storage',control=role,report_sha256=file_sha256(target/'report.json'),
                reclaimed_bytes=row['reclaimed_bytes'],wall_seconds=row['wall_seconds']))
        inode=project_snapshot_inodes(Path.cwd(),6 if role=='one' else 4)
        if os.statvfs(a.root).f_favail<inode['required_free_inodes']:
            target=a.root/(IDENTITY+'-'+role+'-bytecode/report.json')
            command(role+'-closed-bytecode','scripts.reclaim_closed_snapshot_bytecode',
                ['--root',str(a.root),'--output',str(target),'--apply'],240)
            row=read(target)
            if row['status']!='COMPLETE_CLOSED_BYTECODE_RECLAMATION':raise ValueError('Closed-source bytecode maintenance failed')
            maintenance.append(dict(kind='regenerable_closed_bytecode',control=role,report_sha256=file_sha256(target),
                removed_files=row['removed_cache_files'],wall_seconds=row['wall_seconds']))
        if (shutil.disk_usage(a.root).free<47*1024**3
                or os.statvfs(a.root).f_favail<project_snapshot_inodes(Path.cwd(),6 if role=='one' else 4)['required_free_inodes']):
            raise RuntimeError('Bounded permitted maintenance cannot satisfy next admission;no invented training or destructive cleanup')
        write_state(PUBLIC/'maintenance-summary.json',dict(records=maintenance))
    try:
        # The already-owned mixing controller keeps its own original clock.
        # This queue observes it and never replaces or cancels that controller.
        wait_control('mixing')
        for role in plan['remaining_controls']:
            prepare_resources(role)
            command(role+'-registration','scripts.run_lean_baseline_main',
                ['--root',str(a.root),'--profile',role,'--prepare-only'],300)
            parent=a.root/identity(role)/'plan.json';child=read(parent)
            if child['model_variant']!=role or child['seeds']!=plan['seeds'] or child['automatic_4b_admission']:
                raise AssertionError('New registration changed frozen control scope')
            public=Path('results/lean_baseline_controls_20261026/main')/role
            publish([public,PUBLIC], 'Register frozen paired '+role+' control after preceding numerical gates')
            write_state(PUBLIC/'status.json',dict(status='RUNNING_REGISTERED_PAIRED_CONTROL',control=role,maximum_chips=16))
            command(role+'-main','scripts.run_lean_baseline_main',['--root',str(a.root),'--profile',role],14460)
            wait_control(role)
            publish([public,PUBLIC], 'Publish completed audited paired '+role+' control')
        result=dict(status='COMPLETE_ALREADY_DECLARED_LEAN_CONTROLS_QUEUE',completed_controls=records,
            maintenance=maintenance,automatic_4b_admission=False,terminal_success=False)
    except BaseException as error:
        result.update(error=str(error),error_kind=type(error).__name__,traceback=traceback.format_exc(),
            completed_controls=records,maintenance=maintenance,operator_attention_required=True)
        # Write the cause before any owned-child cleanup can obscure it.
        write_state(out/'initiating-failure.json',result);write_state(PUBLIC/'failure-status.json',result)
        raise
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=sha,finished_unix=time.time(),test_accessed=False,terminal_alpha_spent=0)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0,
            operator_attention_required=result.get('operator_attention_required',False)))
        lock_file.close()


if __name__=='__main__':main()
