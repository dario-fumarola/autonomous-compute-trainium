"""Recovery of orchestration state without changing frozen scientific code."""
import hashlib
import json
import os
import signal
import statistics
import time
from pathlib import Path


def verify_saved_sources(repo, plan):
    for name, expected in plan['source_sha256'].items():
        path=repo/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Saved frozen source changed: '+name)
    return {'verified_paths':len(plan['source_sha256']),
            'scope':'Only saved manifest membership is authoritative; unrelated files are ignored.'}


def process_matches(pid, module, output):
    try:
        proc=Path('/proc')/str(pid)
        if proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z':return False
        args=proc.joinpath('cmdline').read_bytes().split(b'\0')
        index=args.index(b'-m')
        return (args[index+1]==module.encode() and b'--output' in args and
                args[args.index(b'--output')+1]==str(output).encode())
    except (FileNotFoundError,PermissionError,ValueError,IndexError):return False


def stop_verified_process(pid, module, output, grace=10):
    if not process_matches(pid,module,output):return {'matching_live_child':False,'terminal':True}
    os.kill(pid,signal.SIGTERM);limit=time.monotonic()+grace
    while process_matches(pid,module,output) and time.monotonic()<limit:time.sleep(.05)
    forced=process_matches(pid,module,output)
    if forced:os.kill(pid,signal.SIGKILL)
    limit=time.monotonic()+2
    while process_matches(pid,module,output) and time.monotonic()<limit:time.sleep(.05)
    terminal=not process_matches(pid,module,output)
    if not terminal:raise RuntimeError('Verified CPU child did not terminate')
    return {'matching_live_child':True,'terminal':terminal,'forced_kill':forced}


class ExistingAudit:
    """Adopt a verified read-only audit handle; do not restart a live child."""
    def __init__(self,pid,module,output):self.pid,self.module,self.output=pid,module,output
    @property
    def returncode(self):
        if process_matches(self.pid,self.module,self.output):return None
        path=self.output/'report.json'
        if path.exists() and json.loads(path.read_text())['status'].startswith('PASS_'):return 0
        return 1
    def poll(self):return self.returncode
    def wait(self,timeout=None):
        deadline=None if timeout is None else time.monotonic()+timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Existing audit wait expired; handle remains live')
            time.sleep(.1)
        return self.returncode
    def terminate(self):return stop_verified_process(self.pid,self.module,self.output)
    def kill(self):return stop_verified_process(self.pid,self.module,self.output,grace=0)


def completed_stage(root,name):
    report=root/'runs'/name/'report.json';closure=root/'snapshots'/name/'closure.json'
    if not report.exists() or not closure.exists():return None
    try:r=json.loads(report.read_text());c=json.loads(closure.read_text())
    except json.JSONDecodeError:return None  # An unclosed writer is pending.
    if c['exit_code']!=0 or not r['status'].startswith('PASS_'):return None
    return r,c


def planned_stages(plan,count,include_longer=False):
    stages={}
    def train(name,arm,start,end):stages[name]={'kind':'train','arm':arm,'targets':count(end)-count(start)}
    def score(name):stages[name]={'kind':'score'}
    for arm in ['ordinary','fixed','mixture']:
        depth=1 if arm=='ordinary' else 2
        for lr in plan['lr_candidates']:
            name=f'vdepth-v1-tune-{arm}-lr{int(lr*1e6)}'
            train(name,arm,0,1152);score(name+f'-dev-r{depth}')
        for seed in plan['seeds']:
            prior=0
            for update in [384,1152,3456,3840]:
                name=f'vdepth-v1-{arm}-s{seed}-u{update}';train(name,arm,prior,update);prior=update
                if update!=3456:score(name+f'-dev-r{depth}')
            if arm=='ordinary':
                for start,end in [(3456,5846),(5846,6496)]:train(f'vdepth-v1-{arm}-s{seed}-u{end}',arm,start,end)
                score(f'vdepth-v1-{arm}-s{seed}-u6496-dev-r1')
            for update in [3840]+([6496] if arm=='ordinary' else []):
                for d in ([1] if arm=='ordinary' else [1,2,3,4]):
                    for split in ['dev','test']:score(f'vdepth-v1-{arm}-s{seed}-u{update}-{split}-r{d}')
        if include_longer:
            seed=plan['seeds'][0];prior=3456
            for update in [7680,10368,11520]:
                name=f'vdepth-v1-long-{arm}-s{seed}-u{update}';train(name,arm,prior,update);prior=update
                score(name+f'-dev-r{depth}')
            for d in ([1] if arm=='ordinary' else [1,2,3,4]):
                for split in ['dev','test']:score(f'vdepth-v1-long-{arm}-s{seed}-u11520-{split}-r{d}')
    return stages


def remaining_budget(plan,root,count):
    state=json.loads((root/'variable-depth-campaign-v2/state.json').read_text()) if (root/'variable-depth-campaign-v2/state.json').exists() else {}
    include_longer=state.get('stages',{}).get('longer_exposure',{}).get('status') in ['RUNNING','COMPLETE']
    stages=planned_stages(plan,count,include_longer);complete={};pending={};samples={a:[] for a in ['ordinary','fixed','mixture']};scoring=[]
    rates=plan['budget']['measured_targets_per_second_per_half_by_arm']
    for name,stage in stages.items():
        done=completed_stage(root,name)
        if done:
            r,c=done;complete[name]=stage
            if stage['kind']=='score':scoring.append(c['wall_seconds'])
            else:samples[stage['arm']].append(max(0,c['wall_seconds']-stage['targets']/rates[stage['arm']]))
        else:pending[name]=stage
    setup={a:statistics.median(v) if v else 180 for a,v in samples.items()}
    score_seconds=statistics.median(scoring) if scoring else 100
    training=sum(s['targets']/rates[s['arm']]+setup[s['arm']] for s in pending.values() if s['kind']=='train')
    readout=sum(s['kind']=='score' for s in pending.values())*score_seconds
    audit_path=root/'variable-depth-campaign-v2/independent-cpu/progress.json'
    audited=json.loads(audit_path.read_text())['records'] if audit_path.exists() else []
    per_audit=max([r['active_seconds'] for r in audited] or [600])
    audits_remaining=max(0,12-len(audited))*per_audit
    if include_longer:
        long_path=root/'variable-depth-campaign-v2/long-independent-cpu/progress.json'
        long_done=len(json.loads(long_path.read_text())['records']) if long_path.exists() else 0
        audits_remaining+=max(0,3-long_done)*per_audit
    return {'complete_stages':len(complete),'pending_stages':list(pending),
            'pending_training_job_seconds':training,'pending_readout_job_seconds':readout,
            'pending_primary_audit_seconds':audits_remaining,
            'remaining_training_wall_seconds_upper':training/2*1.3,
            'remaining_wall_seconds_upper':max((training+readout)/2,audits_remaining)*1.3+300,
            'previously_launched_longer_stage':include_longer,
            'scope':'Completed successful report+closure pairs count once. Pending work only, two eight-chip slots, observed per-arm setup and scoring medians, 30% margin and five-minute closure allowance. Active unclosed stages remain conservatively pending; this is an estimate, not a restart or changed scientific budget.'}


def archive_partial_audit(output):
    progress=output/'progress.json'
    completed={r['identity'] for r in json.loads(progress.read_text())['records']} if progress.exists() else set()
    recovery=output/'recovery'
    archived=[]
    if recovery.exists():
        for path in recovery.iterdir():
            if path.is_dir() and path.name not in completed:
                target=output/'interrupted'/str(time.time_ns())/path.name
                target.parent.mkdir(parents=True);path.rename(target);archived.append(path.name)
    return archived


def live_model_workers():
    result=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=path.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        if b'scripts.train_variable_depth' in args or b'scripts.evaluate_depth_curve' in args:
            result.append(int(path.parent.name))
    return result


def preserve_test_release(proposed,existing):
    if existing is None:return proposed
    def comparable(value):
        return {k:(sorted(v,key=lambda m:m['identity']) if k=='models' else v)
                for k,v in value.items() if k!='frozen_unix'}
    if comparable(proposed)!=comparable(existing):raise ValueError('Existing TEST release cannot change during recovery')
    return existing
