"""Explicit finite follow-on stages; no success merely because a fit exits."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running,metadata,OWNER,cancel_worker_container
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.root/'low-depth-pilot-queue-v2';public=Path('results/low_depth_pilot_queue_20261024/recovery-v2')
    files=['scripts/run_low_depth_pilot_queue_v2.py','scripts/prepare_low_depth_tuning_v2.py',
        'scripts/run_low_depth_tuning_v2.py','scripts/train_low_depth_runtime.py',
        'compute_trainium/parcae_runtime_scalars.py','scripts/prepare_low_depth_main.py',
        'scripts/run_low_depth_main.py','scripts/run_low_depth_gradient_validation.py',
        'scripts/capture_low_depth_gradient.py','scripts/audit_low_depth_gradient.py',
        'compute_trainium/low_depth_gradient_probe.py','docs/LOW_DEPTH_MAIN_PROTOCOL_20261024.md',
        'docs/LOW_DEPTH_TUNING_RUNTIME_AMENDMENT_20261024.md','docs/LOW_DEPTH_MATURE_GRADIENT_POLICY_20261024.md']
    plan_file=out/'plan.json'
    if plan_file.exists():
        if not a.resume:raise FileExistsError('Queue recovery requires explicit --resume')
        plan=json.loads(plan_file.read_text())
    else:
        if a.resume:raise ValueError('No registered queue to resume')
        out.mkdir();public.mkdir(parents=True,exist_ok=False)
        plan={'identity':'low-depth-pilot-queue-v2','registered_unix':time.time(),'deadline_unix':time.time()+14*3600,
            'source_sha256':{f:file_sha256(f) for f in files},
            'stages':['complete changing-schedule numerical/profile qualification',
                'freeze and run runtime-input four-rate screen','finish gradient-capture driver qualification',
                'qualify selected301M gradients','freeze and run one-seed matched-work/equal-data pilot'],
            'test_accessed':False,'new_terminal_attempt':False,
            'scope':'Fixed approved finite stages; every child ends in verified success or failed prerequisite. No gate relaxation,window reset,budget expansion,qualification/tuning-weight reuse,fresh confirmation or automatic significant-result claim. Each child retains independent cap/receipts.'}
        write_state(plan_file,plan);write_state(public/'plan.json',plan)
    if (out/'report.json').exists() and json.loads((out/'report.json').read_text())['status']=='COMPLETE_REGISTERED_LOW_DEPTH_PILOT_QUEUE':return
    child=None;records=[]
    def frozen():
        if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Registered next-stage source changed')
        if time.time()>plan['deadline_unix']:raise TimeoutError('Original finite next-stage queue window expired')
    def status(stage,state):write_state(public/'status.json',{'status':state,'stage':stage,'observed_unix':time.time(),'controller_assigned_chips':0})
    def wait_report(stage,path,required):
        status(stage,'WAITING_FOR_REGISTERED_PREREQUISITE')
        while not path.exists():frozen();time.sleep(5)
        row=json.loads(path.read_text())
        if row['status']!=required:raise RuntimeError('Failed prerequisite '+stage+': '+row['status'])
        records.append({'stage':stage,'status':row['status'],'receipt_sha256':file_sha256(path)})
        return row
    def run(stage,module,args,seconds):
        nonlocal child
        frozen();status(stage,'RUNNING_REGISTERED_STAGE');begin=time.time();limit=min(plan['deadline_unix'],begin+seconds)
        env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (out/(stage+f'-{int(begin)}.log')).open('x') as log:
            child=subprocess.Popen([os.sys.executable,'-u','-m',module,*args],env=env,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        write_state(out/'active-child.json',{'stage':stage,'pid':child.pid,'started_unix':begin,'limit_unix':limit})
        while child.poll() is None:
            frozen()
            if time.time()>limit:raise TimeoutError('Finite child stage deadline: '+stage)
            time.sleep(5)
        code=child.returncode;child=None
        records.append({'stage':stage,'exit_code':code,'wall_seconds':time.time()-begin})
        write_state(public/'progress.json',{'records':records})
        if code:raise RuntimeError('Registered child failed: '+stage)
    result={'status':'STOPPED_REGISTERED_LOW_DEPTH_PILOT_QUEUE'}
    try:
        frozen()
        wait_report('runtime-qualification',Path('results/parcae_runtime_scalars_20261024/recovery-v2/report.json'),
                    'PASS_COMPLETE_PARCAE_RUNTIME_SCALAR_QUALIFICATION')
        experiment=a.root/'parcae-low-depth-pilot-v1'
        if not (experiment/'tuning-plan-v2.json').exists():
            run('prepare-tuning','scripts.prepare_low_depth_tuning_v2',['--root',str(a.root)],120)
        tune_report=Path('results/parcae_low_depth_pilot_20261024/tuning-v2/report.json')
        if not tune_report.exists():
            args=['--root',str(a.root)]
            if (experiment/'tuning-execution-v2/execution-start.json').exists():args.append('--resume')
            run('run-tuning','scripts.run_low_depth_tuning_v2',args,10890)
        tune=wait_report('tuning-complete',tune_report,'COMPLETE_FROZEN_LOW_DEPTH_LR_SCREEN')
        wait_report('probe-driver',Path('results/low-depth-gradient-driver-v1/report.json'),'PASS_LOW_DEPTH_GRADIENT_VALIDATION')
        selected_report=Path('results/low-depth-gradient-selected-v1/report.json')
        if not selected_report.exists():
            points={}
            for arm,row in tune['selected'].items():
                trial=next(r for r in tune['trials'] if r['case_id']==row['case_id'])
                points[arm]={'case_id':row['case_id'],'checkpoint':f'runs/{trial["final_identity"]}/checkpoint-u576.pt',
                             'checkpoint_sha256':row['checkpoint_sha256']}
            point_file=out/'selected-points.json'
            if point_file.exists() and json.loads(point_file.read_text())!=points:raise AssertionError('Selected point registration changed')
            write_state(point_file,points)
            args=['--root',str(a.root),'--identity','low-depth-gradient-selected-v1']
            if not (a.root/'low-depth-gradient-selected-v1/plan.json').exists():
                args+=['--parent-plan',str(experiment/'tuning-plan-v2.json'),'--points-manifest',str(point_file)]
            run('selected-gradient-validation','scripts.run_low_depth_gradient_validation',args,2790)
        wait_report('selected-gradients',selected_report,'PASS_LOW_DEPTH_GRADIENT_VALIDATION')
        if not (experiment/'main-plan.json').exists():run('prepare-main','scripts.prepare_low_depth_main',['--root',str(a.root)],120)
        main_report=Path('results/parcae_low_depth_pilot_20261024/main/report.json')
        if not main_report.exists():
            args=['--root',str(a.root)]
            if (experiment/'main-execution/execution-start.json').exists():args.append('--resume')
            run('run-main','scripts.run_low_depth_main',args,28890)
        wait_report('main-complete',main_report,'COMPLETE_FROZEN_LOW_DEPTH_MAIN_PILOT')
        result={'status':'COMPLETE_REGISTERED_LOW_DEPTH_PILOT_QUEUE','research_goal_achieved':False,
                'next':'Interpret one-seed DEV pilot and prespecified mechanism diagnostics; terminal fresh replication is a separate scientific decision.'}
    except BaseException as error:result.update(error=str(error));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                for container in managed_running():
                    data=metadata(container);owner=(data['Config'].get('Labels') or {}).get(OWNER,'')
                    if owner.startswith(('low-depth-tune-v2-','low-depth-main-v1-','low-depth-main-gradient-v1-',
                                         'low-depth-gradient-selected-v1-')):
                        cancel_worker_container(container,owner,'finite_pilot_queue_stop')
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
        result.update(records=records,wall_seconds=time.time()-plan['registered_unix'],
            test_accessed=False,new_terminal_attempt=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        status('terminal',result['status'])


if __name__=='__main__':main()
