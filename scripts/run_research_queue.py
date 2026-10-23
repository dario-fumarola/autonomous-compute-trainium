"""Persistent finite queue: detached stages, dependency failures and resume receipts."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from compute_trainium.queue import prerequisites,write_state
from compute_trainium.training import file_sha256


def alive(pid):
    try:
        os.kill(pid,0)
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
        return stat!='Z'
    except ProcessLookupError:return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());a.output.mkdir(exist_ok=True)
    state_path=a.output/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'plan_sha256':file_sha256(a.plan),'stages':{}}
    if state['plan_sha256']!=file_sha256(a.plan):raise ValueError('Queue plan changed')
    def save():state['heartbeat_unix']=time.time();write_state(state_path,state)
    for stage in plan['stages']:
        name=stage['name'];current=state['stages'].get(name,{})
        if current.get('status') in ['COMPLETE','FAILED','BLOCKED','DEADLINE']:continue
        ready,deps=prerequisites(stage,state['stages'])
        if ready!='READY':state['stages'][name]={'status':'BLOCKED','prerequisites':deps};save();continue
        expected=Path(stage['report']);pid=current.get('pid')
        if not expected.exists() and not pid:
            if time.time()>plan['deadline_unix']:
                state['stages'][name]={'status':'DEADLINE'};save();continue
            if stage.get('wait_pid'):
                pid=stage['wait_pid'];started=time.time()
            else:
                log=(a.output/(name+'.log')).open('a')
                process=subprocess.Popen(stage['command'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,**stage.get('environment',{})),start_new_session=True)
                pid=process.pid;started=time.time();log.close()
            current={'status':'RUNNING','pid':pid,'started_unix':started};state['stages'][name]=current;save()
        while not expected.exists() and pid and alive(pid):
            if time.time()>min(plan['deadline_unix'],current['started_unix']+stage['timeout_seconds']):
                if not stage.get('wait_pid'):
                    try:os.killpg(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                current.update(status='DEADLINE',ended_unix=time.time());save();break
            save();time.sleep(5)
        if expected.exists():
            result=json.loads(expected.read_text());success=result.get('status') in stage['accepted_statuses']
            current.update(status='COMPLETE' if success else 'FAILED',result_status=result.get('status'),report_sha256=file_sha256(expected),ended_unix=time.time())
        elif current.get('status')!='DEADLINE':current.update(status='FAILED',reason='Stage process ended without its required result',ended_unix=time.time())
        state['stages'][name]=current;save();print(json.dumps({'stage':name,**current}),flush=True)
    state['status']='COMPLETE_QUEUE' if all(v['status']=='COMPLETE' for v in state['stages'].values()) else 'QUEUE_FINISHED_WITH_EXPLICIT_FAILURES'
    save();write_state(a.output/'report.json',state)


if __name__=='__main__':main()
