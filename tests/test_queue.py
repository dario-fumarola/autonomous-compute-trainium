from compute_trainium.queue import prerequisites,write_state
import json


def test_dependency_failure_is_explicit_and_independent_stages_continue(tmp_path):
    states={'a':{'status':'FAILED'}}
    assert prerequisites({'dependencies':['a']},states)==('BLOCKED',['a'])
    assert prerequisites({'dependencies':[]},states)==('READY',[])
    assert prerequisites({'dependencies':['b']},states)==('WAITING',[])
    states['b']={'status':'COMPLETE'}
    assert prerequisites({'dependencies':['b']},states)==('READY',[])
    p=tmp_path/'queue.json';write_state(p,states);assert json.loads(p.read_text())==states
    assert not p.with_suffix('.tmp').exists()


def test_process_failure_is_recorded_resume_does_not_repeat_completed_stage(tmp_path):
    import subprocess,sys,time
    failed=tmp_path/'failed.json';passed=tmp_path/'passed.json';counter=tmp_path/'counter'
    plan={'deadline_unix':time.time()+60,'stages':[
        {'name':'failed','command':[sys.executable,'-c','raise SystemExit(3)'],'report':str(failed),'accepted_statuses':['PASS'],'timeout_seconds':15},
        {'name':'independent','command':[sys.executable,'-c',f"from pathlib import Path; Path({str(passed)!r}).write_text('{{\"status\":\"PASS\"}}'); Path({str(counter)!r}).write_text('once')"], 'report':str(passed),'accepted_statuses':['PASS'],'timeout_seconds':15},
        {'name':'dependent','dependencies':['failed'],'command':[sys.executable,'-c','raise SystemExit(99)'],'report':str(tmp_path/'blocked.json'),'accepted_statuses':['PASS'],'timeout_seconds':15}]}
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan));output=tmp_path/'queue'
    command=[sys.executable,'-m','scripts.run_research_queue','--plan',str(p),'--output',str(output)]
    subprocess.run(command,check=True,capture_output=True,timeout=40)
    state=json.loads((output/'report.json').read_text())
    assert {k:v['status'] for k,v in state['stages'].items()}=={'failed':'FAILED','independent':'COMPLETE','dependent':'BLOCKED'}
    subprocess.run(command,check=True,capture_output=True,timeout=15)
    assert counter.read_text()=='once'
