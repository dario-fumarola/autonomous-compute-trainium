"""Close remaining original opportunities after a numerical tuning failure.

No retry of the failed generated fit, additional LR, renewed deadline or repair
of the original gate. Finite endpoints remain subject to their full audits.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution

ID = 'chunked-control-completion-v1'
ORIGINAL = 'chunked-persistent-screen-v1'
PUBLIC = Path('results/chunked_persistent_workspace_20261029/control-completion-v1')
FAILED = 'generated-lr2'


def read(p):
    return json.loads(p.read_text())


def nonfinite_failure(root, phase):
    closure = read(root/'snapshots'/phase/'closure.json')
    supervisor = read(root/'logs'/(phase+'.supervisor.json'))
    log = (root/'logs'/(phase+'.log')).read_text()
    return (closure['exit_code'] == 1 and supervisor['worker_exit_code'] == 1
            and supervisor['supervisor_failure'] is None
            and supervisor['container_oom_kill_delta'] == 0
            and 'The total norm of order 2.0 for gradients' in log
            and 'is non-finite, so it cannot be clipped' in log)


def prepare(root):
    oldpath = root/ORIGINAL/'plan.json'; old = read(oldpath)
    oldreport = root/ORIGINAL/'report.json'; stopped = read(oldreport)
    if stopped['status'] != 'STOPPED_CHUNKED_PERSISTENT_SCREEN' or managed_running():
        raise AssertionError('Original stop and closed owner required')
    if not nonfinite_failure(root,old['arms'][FAILED]['phase']):
        raise AssertionError('Only the observed non-finite tuning failure is covered')
    if any(file_sha256(f)!=h for f,h in old['source_sha256'].items()):
        raise AssertionError('Original scientific sources must remain unchanged')
    completed = [a for a in old['arms'] if a in stopped['records']]
    if set(completed) != {'ordinary-lr0','ordinary-lr1','ordinary-lr2','generated-lr0','generated-lr1'}:
        raise AssertionError('Exactly the five complete original histories required')
    for arm in completed:
        if stopped['records']['audit-'+arm]['status'] != 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':
            raise AssertionError('Original complete mature audits required')
    remaining = [f'constant-lr{i}' for i in range(3)]
    if any((root/'runs'/old['arms'][a]['phase']).exists() for a in remaining):
        raise AssertionError('Only previously unstarted original opportunities')
    if (root/ORIGINAL/'select').exists() or (root/'runs'/(ORIGINAL+'-evaluation')).exists():
        raise AssertionError('Register before any quality selection')
    if min(shutil.disk_usage(root).free-4*1024**3,shutil.disk_usage('/').free)<35*1024**3:
        raise RuntimeError('Four GiB completion allowance above protected reserves')
    projection = project_snapshot_inodes(Path('.'),4)
    if os.statvfs(root).f_favail < projection['required_free_inodes']:
        raise RuntimeError('Remaining snapshots must fit protected inode reserve')
    now=time.time()
    if old['absolute_deadline_unix']-now<7200:
        raise TimeoutError('Two-hour measured completion allowance must fit original deadline')
    p=copy.deepcopy(old);p['identity']=ID
    p['arms']={a:copy.deepcopy(v) for a,v in old['arms'].items() if a!=FAILED}
    for arm in remaining:p['arms'][arm]['phase']=ID+'-'+arm
    names=[p['arms'][a]['phase'] for a in remaining]+[ID+'-evaluation']
    p.update(registered_unix=now,seconds_limit=old['absolute_deadline_unix']-now,
        original_plan_sha256=file_sha256(oldpath),original_report_sha256=file_sha256(oldreport),
        original_status=stopped['status'],original_gate_status='UNMET: all-nine-complete condition failed',
        original_costs_sha256=file_sha256(Path('results/chunked_persistent_workspace_20261029/learning-v1/execution-costs.json')),
        completed_original_arms=completed,remaining_original_opportunities=remaining,
        unavailable_arms={FAILED:dict(lr=.003,reason='Non-finite gradient after 80 persisted updates; no mature endpoint, no retry, no invented score.')},
        root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},
        inode_projection=projection,artifact_allowance_bytes=4*1024**3,
        evaluation_points=[x for x in p['evaluation_points'] if x['arm']!=FAILED],
        scope='Post-stop completion of the original three unstarted constant-slot opportunities and descriptive comparison of audited endpoints. Original gate stays unmet. No extra LR, retry, new source, longer endpoint, automatic promotion or renewed deadline. Failed fits have no NLL and are excluded from selection, with LR bracketing marked incomplete.')
    for f in ('scripts/complete_chunked_controls.py','scripts/analyze_chunked_completion.py',
              'tests/test_chunked_completion.py','docs/CHUNKED_CONTROL_COMPLETION_20261029.md'):
        p['source_sha256'][f]=file_sha256(f)
        p['host_only_source_sha256'][f]=p['source_sha256'][f]
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    for arm in completed:
        dest=out/('audit-'+arm);dest.mkdir()
        original=root/ORIGINAL/('audit-'+arm)/'report.json'
        shutil.copyfile(original,dest/'report.json')
        write_state(dest/'provenance.json',dict(original_sha256=file_sha256(original),reused=True,new_updates=0))


def run(root):
    out=root/ID;p=read(out/'plan.json')
    if file_sha256(root/ORIGINAL/'report.json')!=p['original_report_sha256']:
        raise AssertionError('Original failed result must remain intact')
    e=ScreenExecution(root,out/'plan.json',out/'fit-execution');evaluation=None
    original=read(root/ORIGINAL/'report.json')
    result=dict(status='RUNNING_CHUNKED_CONTROL_COMPLETION',records={k:v for k,v in original['records'].items()},
                unavailable=copy.deepcopy(p['unavailable_arms']),original_gate_status=p['original_gate_status'])
    def stop(number,frame):raise KeyboardInterrupt('Bounded completion owner interrupted')
    handlers={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for arm in p['remaining_original_opportunities']:
            phase=p['arms'][arm]['phase']
            try:
                fit=e.native(phase,'scripts.chunked_persistent_scientific_entry',
                    ['--entry','train','--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json',
                     '--plan-sha',e.sha,'--arm',arm,'--comm-port',str(p['root_comm_ports'][phase])],
                    'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS')
            except RuntimeError:
                # A registered failed tuning opportunity is excluded, never retried.
                # Resource, supervisor, cleanup and audit failures still stop everything.
                if e.cleanup_errors or managed_running() or not nonfinite_failure(root,phase):raise
                result['unavailable'][arm]=dict(lr=p['arms'][arm]['lr'],reason='Non-finite gradient; no endpoint, score or retry.')
                e.failed.clear();e.check()
                write_state(PUBLIC/'progress.json',result);continue
            result['records'][arm]=fit
            dest=out/('audit-'+arm)
            result['records']['audit-'+arm]=e.cpu(ID+'-audit-'+arm,'scripts.chunked_persistent_scientific_entry',
                ['--entry','scientific-audit','--root',str(root),'--plan',str(out/'plan.json'),
                 '--plan-sha',e.sha,'--arm',arm,'--output',str(dest)],dest/'report.json','PASS_INTERLEAVED_SCREEN_MATURE_AUDITS')
            write_state(PUBLIC/'progress.json',result)
        eligible={a:v for a,v in p['arms'].items() if a not in result['unavailable']}
        if any(not any(v['family']==f for v in eligible.values()) for f in ('ordinary','generated','constant')):
            raise RuntimeError('No complete audited candidate in a family')
        digests={result['records'][a]['initial_model_digest'] for a in eligible}
        if len(digests)!=1:raise AssertionError('Exact paired original initialization required')
        readplan=copy.deepcopy(p);readplan['arms']=eligible
        readplan['unavailable_arms']=result['unavailable']
        readplan['evaluation_points']=[x for x in p['evaluation_points'] if x['arm'] in eligible]
        name=ID+'-evaluation';readplan['root_comm_ports']={name:p['root_comm_ports'][name]}
        readplan['phases']={name:{}}
        write_state(out/'readout-plan.json',readplan);write_state(PUBLIC/'readout-plan.json',readplan)
        e.close(PUBLIC/'fit-execution-costs.json');e=None
        evaluation=ScreenExecution(root,out/'readout-plan.json',out/'readout-execution')
        result['records']['evaluation']=evaluation.native(name,'scripts.chunked_persistent_scientific_entry',
            ['--entry','evaluate','--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/readout-plan.json',
             '--plan-sha',evaluation.sha,'--comm-port',str(p['root_comm_ports'][name])],
            'COMPLETE_INTERLEAVED_SCREEN_EVALUATION')
        for action,status in [('select','COMPLETE_CHUNKED_COMPLETION_SELECTION'),
                              ('readout','COMPLETE_CHUNKED_PERSISTENT_EXACT_READOUT'),
                              ('compare','COMPLETE_CHUNKED_CONTROL_COMPARISON')]:
            dest=out/action
            module='scripts.readout_chunked_persistent' if action=='readout' else 'scripts.analyze_chunked_completion'
            args=['--root',str(root),'--plan',str(out/'readout-plan.json'),'--output',str(dest)]
            if action!='readout':args+=['--action',action]
            result['records'][action]=evaluation.cpu(ID+'-'+action,module,args,dest/'report.json',status)
            write_state(PUBLIC/(action+'.json'),result['records'][action])
        result['status']='COMPLETE_CHUNKED_CONTROL_COMPLETION'
    except BaseException as error:
        result.update(status='STOPPED_CHUNKED_CONTROL_COMPLETION',error_type=type(error).__name__,error=str(error));raise
    finally:
        if e:e.close(PUBLIC/'fit-execution-costs.json')
        if evaluation:evaluation.close(PUBLIC/'readout-execution-costs.json')
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--action',choices=['prepare','run'],required=True)
    args=parser.parse_args();(prepare if args.action=='prepare' else run)(args.root)
