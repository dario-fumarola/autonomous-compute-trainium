"""Adopt the surviving supervised fit after a chat-server restart.

No graph, optimizer, fit duration, endpoint, source order or deadline changes.
This host-only controller is run as a bounded user service, with no auto retry.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

from compute_trainium.execution_isolation import managed_running, metadata, verify_owner, cancel_worker_container
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution
import scripts.run_chunked_persistent_screen as original

ID='chunked-persistent-controller-recovery-v1'
PUBLIC=Path('results/chunked_persistent_workspace_20261029/controller-recovery-v1')


def validate_launch(launch,plan,sha,module,options):
    if launch['entry_module']!=module or launch['arguments']!=options:
        raise AssertionError('Adoption must retain exact original module and arguments')
    if launch['physical_chips']!=list(range(16)) or sha not in options:
        raise AssertionError('All chips and exact original plan required')
    # The launch manifest inventories Python only; the actual snapshot check
    # below also verifies the pinned shell launcher and any other worker file.
    if any(launch['source_sha256'].get(f)!=h for f,h in plan['worker_source_sha256'].items() if f.endswith('.py')):
        raise AssertionError('Surviving snapshot differs from frozen worker source')


def validate_snapshot(snapshot,plan):
    if any(file_sha256(snapshot/f)!=h for f,h in plan['worker_source_sha256'].items()):
        raise AssertionError('Actual worker snapshot bytes changed')


def prepare(root):
    folder=root/original.ID;plan_file=folder/'plan.json';plan=json.loads(plan_file.read_text())
    with (folder/'execution/controller.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if any(file_sha256(f)!=h for f,h in plan['source_sha256'].items()):
        raise AssertionError('Original sources must remain unchanged')
    phase=plan['arms']['ordinary-lr0']['phase'];owners=managed_running()
    if len(owners)!=1:raise AssertionError('Exactly the surviving original fit required')
    verify_owner(metadata(owners[0]),phase)
    launch_file=root/'snapshots'/phase/'launch.json';launch=json.loads(launch_file.read_text())
    options=['--entry','train','--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{original.ID}/plan.json',
        '--plan-sha',file_sha256(plan_file),'--arm','ordinary-lr0','--comm-port',str(plan['root_comm_ports'][phase])]
    validate_launch(launch,plan,file_sha256(plan_file),'scripts.chunked_persistent_scientific_entry',options)
    validate_snapshot(launch_file.parent,plan)
    now=time.time()
    if now>=plan['absolute_deadline_unix']:raise TimeoutError('Original deadline remains binding')
    receipt=dict(identity=ID,registered_unix=now,reason='Chat-server restart interrupted only the host controller; original supervised all-chip fit survives.',
        scientific_plan_sha256=file_sha256(plan_file),original_execution_sha256=file_sha256(folder/'execution/execution-start.json'),
        original_absolute_deadline_unix=plan['absolute_deadline_unix'],adopted_phase=phase,
        adopted_launch_sha256=file_sha256(launch_file),new_fit_updates=0,new_scientific_budget=0,
        source_sha256={f:file_sha256(f) for f in ['scripts/resume_chunked_persistent_controller.py','tests/test_chunked_controller_recovery.py']},
        service_restart_policy='no',scope='Join the existing supervised fit, require its successful closure and audit, then execute only the remaining frozen original queue. Never relaunch a partial fit or renew its deadline. All original costs remain charged.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'registration.json',receipt);write_state(PUBLIC/'registration.json',receipt)


class ResumedExecution(ScreenExecution):
    def __init__(self,root,plan_file,out):
        super().__init__(root,plan_file,out,resume=True)
        self.admission=json.loads((self.root/ID/'registration.json').read_text())
        if self.sha!=self.admission['scientific_plan_sha256'] or any(file_sha256(f)!=h for f,h in self.admission['source_sha256'].items()):
            raise AssertionError('Frozen host adoption or original scientific plan changed')

    def native(self,identity,module,options,status):
        directory=self.root/'runs'/identity
        if not directory.exists():return super().native(identity,module,options,status)
        snapshot=self.root/'snapshots'/identity
        launch=json.loads((snapshot/'launch.json').read_text())
        validate_launch(launch,self.plan,self.sha,module,options)
        validate_snapshot(snapshot,self.plan)
        container=None
        try:
            matching=[]
            for owner in managed_running():
                data=metadata(owner)
                if (data['Config'].get('Labels') or {}).get('ac-trainium-owner')!=identity:
                    raise AssertionError('An unrelated owner cannot be adopted or cancelled')
                verify_owner(data,identity);matching.append(owner)
            if len(matching)>1:raise AssertionError('Ambiguous live owner')
            if matching:
                if identity!=self.admission['adopted_phase']:raise AssertionError('Only registered surviving phase may be adopted live')
                container=matching[0];self.native_active[identity]=container
            supervisor=self.root/'logs'/(identity+'.supervisor.json')
            if not supervisor.exists() and container is None:raise RuntimeError('Missing original live supervisor/closure; no partial retry')
            while not supervisor.exists():
                self.check()
                if time.time()>launch['started_unix']+self.native_seconds_limit(identity):
                    raise TimeoutError('Original native phase cap')
                write_state(PUBLIC/'progress.json',dict(status='WATCHING_ORIGINAL_SUPERVISOR',phase=identity,checked_unix=time.time()))
                time.sleep(2)
            s=json.loads(supervisor.read_text());closure=json.loads((snapshot/'closure.json').read_text())
            if s['worker_exit_code'] or s['supervisor_failure'] or s['container_oom_kill_delta'] or closure['exit_code']:
                raise AssertionError('Adopted/reused phase must have a clean original closure')
            result=json.loads((directory/'report.json').read_text())
            if result['status']!=status or result['world']!=64:raise AssertionError('Incomplete original phase')
            write_state(PUBLIC/(identity+'-closure.json'),dict(status='REUSED_ORIGINAL_COMPLETE_PHASE',
                phase=identity,report_sha256=file_sha256(directory/'report.json'),
                launch_sha256=file_sha256(snapshot/'launch.json'),closure_sha256=file_sha256(snapshot/'closure.json'),
                added_optimizer_updates=0))
            return result
        finally:
            if container:
                cancel_worker_container(container,identity,'adopted_original_phase_complete')
                self.native_active.pop(identity,None)


def run(root):
    original.ScreenExecution=ResumedExecution
    original.run(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run'],required=True);a=p.parse_args()
    (prepare if a.action=='prepare' else run)(a.root)
