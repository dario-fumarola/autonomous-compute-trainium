"""Resume only unstarted replay/audit after bounded host admission stop."""
import argparse,json,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_probability_screen import ScreenExecution

ID='interleaved-mixed-round-qualification-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/mixed-round-qualification-v1')


def main(root):
    if managed_running():raise RuntimeError('Previous owner must close')
    old=json.loads((PUBLIC/'report.json').read_text());plan=root/ID/'plan.json';p=json.loads(plan.read_text())
    if old['status']!='STOPPED_INTERLEAVED_ACTUAL_SHAPE' or old.get('error')!='Host allowance unavailable after bounded180-second admission wait':
        raise AssertionError('Only named prelaunch admission failure may recover')
    if time.time()+1200>p['absolute_deadline_unix']:raise RuntimeError('Original deadline must admit full recovery')
    warm=root/'runs'/(ID+'-warm');cost=json.loads((PUBLIC/'execution-costs.json').read_text())
    if len(cost['native_attempts'])!=1 or cost['native_attempts'][0]['exit_code']!=0:
        raise AssertionError('Completed warm phase only')
    if (root/'runs'/(ID+'-replay')).exists():raise AssertionError('Replay must not have started')
    if old['records']['warm']['status']!='COMPLETE_INTERLEAVED_WARM_PENDING_GLOBAL_AUDIT':raise AssertionError('Completed warm record required')
    recovery=PUBLIC/'recovery-v1';recovery.mkdir();private=root/ID/'recovery-v1';private.mkdir()
    write_state(recovery/'prior-stop.json',old);write_state(recovery/'prior-costs.json',cost)
    registration=dict(status='REGISTERED_ADMISSION_RECOVERY',registered_unix=time.time(),
        original_deadline_unix=p['absolute_deadline_unix'],plan_sha256=file_sha256(plan),
        host_recovery_source_sha256=file_sha256(__file__),warm_files={n:file_sha256(warm/n) for n in ('PRIVATE-u103.pt','PRIVATE-u104.pt','PRIVATE-mature-gradient.pt','report.json')},
        scope='Reuse completed warm qualification; run only unstarted exact replay and independent mature audit. No new scientific updates, no source/optimizer/RNG/gate/deadline changes. Original stop and cost preserved.')
    write_state(recovery/'plan.json',registration);write_state(private/'plan.json',registration)
    e=ScreenExecution(root,plan,private/'execution');r=dict(status='RUNNING_RECOVERED_MIXED_QUALIFICATION',records={'warm':old['records']['warm']},recovery=registration)
    try:
        for n,h in registration['warm_files'].items():
            if file_sha256(warm/n)!=h:raise AssertionError('Preserved warm bytes changed')
        name=ID+'-replay';args=['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
            '--plan-sha',e.sha,'--mode','replay','--comm-port',str(p['root_comm_ports'][name]),'--parent',f'/work/instance-nvme/runs/{ID}-warm']
        r['records']['replay']=e.native(name,'scripts.qualify_interleaved_grouped',args,'PASS_INTERLEAVED_WARM_EXACT_REPLAY')
        write_state(recovery/'progress.json',r)
        dest=root/ID/'cpu-audit'
        r['records']['audit']=e.cpu(ID+'-audit','scripts.audit_interleaved_warm',
            ['--root',str(root),'--plan',str(plan),'--plan-sha',e.sha,'--output',str(dest)],dest/'report.json','PASS_INTERLEAVED_MATURE_GLOBAL_CPU')
        r['status']='PASS_INTERLEAVED_ACTUAL_SHAPE_AND_REPLAY'
    except BaseException as error:
        r.update(status='STOPPED_RECOVERED_MIXED_QUALIFICATION',error_type=type(error).__name__,error=str(error));raise
    finally:
        write_state(recovery/'report.json',r);e.close(recovery/'execution-costs.json')
        write_state(PUBLIC/'report.json',r)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);main(a.parse_args().root)
