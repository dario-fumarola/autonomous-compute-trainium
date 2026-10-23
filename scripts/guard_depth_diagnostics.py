"""External lifecycle repair without modifying the already-running annex."""
import argparse
import json
import time
from pathlib import Path

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.recovery_support import process_matches,stop_verified_process


def matching_audits(output):
    result=[]
    for path in Path('/proc').glob('[0-9]*'):
        if process_matches(int(path.name),'scripts.audit_depth_mechanism',output):result.append(int(path.name))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--annex',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--public-output',type=Path,default=Path('results/variable_depth_20261023/readonly-lifecycle-guard.json'))
    a=p.parse_args();a.output.mkdir(exist_ok=False);started=time.time()
    spec=json.loads((a.annex/'specification.json').read_text());deadline=spec['deadline_unix']
    parent='scripts.run_variable_depth_diagnostics';forced_parent=False
    write_state(a.output/'plan.json',{'purpose':'ANNEX_LIFECYCLE_ONLY','deadline_unix':deadline,
        'original_annex_spec_sha256':file_sha256(a.annex/'specification.json'),
        'source_sha256':{name:file_sha256(name) for name in ['scripts/guard_depth_diagnostics.py','scripts/recovery_support.py']},
        'scope':'No scientific source change, model choice, scoring or native launch. Observe the specific annex handle; clean only verified read-only CPU children and preserve/repair terminal reporting.'})
    while process_matches(a.pid,parent,a.annex):
        if time.time()>deadline+60:
            stop_verified_process(a.pid,parent,a.annex);forced_parent=True;break
        write_state(a.output/'state.json',{'status':'WATCHING_VERIFIED_ANNEX','checked_unix':time.time()})
        time.sleep(2)
    children=matching_audits(a.annex/'independent-mechanism');cleanup=[]
    for pid in children:cleanup.append(stop_verified_process(pid,'scripts.audit_depth_mechanism',a.annex/'independent-mechanism'))
    if matching_audits(a.annex/'independent-mechanism'):raise RuntimeError('Read-only CPU child remains live')
    terminal=a.annex/'report.json';repaired=not terminal.exists()
    if repaired:
        write_state(terminal,{'status':'STOPPED_DIAGNOSTIC_CONTROLLER_WITHOUT_REPORT',
            'reason':'Verified annex handle exited or exceeded its deadline without a terminal report',
            'verified_cpu_children_cleaned':len(children),'partial_native_receipts_preserved':True,
            'scope':'No completed diagnostic is inferred. Existing native supervisors retain their own timeouts; no native retry or model change is authorized by this receipt.'})
    report={'status':'REPAIRED_MISSING_DIAGNOSTIC_TERMINAL_REPORT' if repaired else 'PASS_DIAGNOSTIC_LIFECYCLE_GUARD',
            'annex_status':json.loads(terminal.read_text())['status'],'cpu_child_cleanup':cleanup,
            'forced_parent_at_deadline':forced_parent,'wall_seconds_including_waits':time.time()-started,
            'scope':'Lifecycle only. Original diagnostic source/specification preserved; no numerical result or completed scoring claim added.'}
    write_state(a.output/'report.json',report)
    write_state(a.public_output,report)


if __name__=='__main__':main()
