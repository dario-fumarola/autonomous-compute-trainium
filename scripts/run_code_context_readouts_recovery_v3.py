"""Execution-only recovery:explicit recovered endpoint gradient receipts.

The old missing-path failure,original panel/source hashes and absolute deadline
remain preserved. No model/selection/scoring/criterion changes.
"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    selection=a.root/'natural-code-context-v1';out=selection/'evaluation-execution-recovery-v3'
    public=Path('results/natural_code_context_20261024/readouts/recovery-v3');plan_file=out/'plan.json'
    if plan_file.exists():
        if not a.resume:raise FileExistsError('Explicit context-readout recovery required')
        plan=json.loads(plan_file.read_text())
    else:
        if a.resume:raise ValueError('No context-readout execution to recover')
        predecessor=selection/'evaluation-execution-recovery-v2/plan.json'
        plan=json.loads(predecessor.read_text())
        if any(file_sha256(f)!=sha for f,sha in plan['software_sha256'].items()):raise AssertionError('Original context preregistration source changed')
        old_result=json.loads((selection/'evaluation-execution-recovery-v2/report.json').read_text())
        if old_result.get('records'):raise ValueError('This recovery carries an unscored original registration only')
        plan={**plan,'identity':'natural-code-context-readouts-recovery-v3',
            'recovery_registered_unix':time.time(),'recovery_of_plan_sha256':file_sha256(predecessor),
            'software_sha256':{**plan['software_sha256'],'scripts/run_code_context_readouts_recovery_v3.py':file_sha256(__file__),
                'scripts/evaluate_natural_code_context_recovered.py':file_sha256('scripts/evaluate_natural_code_context_recovered.py'),
                'tests/test_context_gradient_receipt_recovery.py':file_sha256('tests/test_context_gradient_receipt_recovery.py')}}
        out.mkdir();public.mkdir(parents=True,exist_ok=False);write_state(plan_file,plan);write_state(public/'plan.json',plan)
    if (out/'report.json').exists() and json.loads((out/'report.json').read_text())['status']=='COMPLETE_FROZEN_CODE_CONTEXT_READOUTS':return
    started=time.time();child=None;records=[];costs=[];cost_row=None
    def guard():
        if time.time()>plan['deadline_unix']:raise TimeoutError('Original context-readout window expired')
        if any(file_sha256(f)!=sha for f,sha in plan['software_sha256'].items()):raise AssertionError('Frozen context scoring source changed')
        for file,key in [('panel.json','selection_panel_sha256'),('scoring-panel-v3.json','scoring_panel_sha256'),('intervention-qualification-v3.json','intervention_qualification_sha256')]:
            if file_sha256(selection/file)!=plan[key]:raise AssertionError('Frozen context panel/qualification changed')
    def prerequisite_failed():
        for file in [Path('results/parcae_low_depth_pilot_20261024/main-recovery-v2/report.json')]:
            if file.exists() and json.loads(file.read_text())['status'].startswith('STOPPED_'):
                raise RuntimeError('Context readout prerequisite stopped: '+str(file))
    result={'status':'STOPPED_FROZEN_CODE_CONTEXT_READOUTS'}
    try:
        main_plan=a.root/'parcae-low-depth-pilot-v1/main-plan-recovery-v2.json'
        for readout in plan['readouts']:
            write_state(public/'status.json',{'status':'WAITING_FOR_EXACT_CANDIDATE_ENDPOINT_GRADIENT_GATE','readout':readout,'assigned_chips':0})
            while True:
                guard();prerequisite_failed()
                path=Path('results/parcae_low_depth_pilot_20261024/main-recovery-v2/gradient-progress.json')
                gradients=json.loads(path.read_text())['records'] if path.exists() else []
                matching=[r for r in gradients if r['arm']=='poisson2' and r['readout']==readout]
                if main_plan.exists() and matching:break
                time.sleep(10)
            parent=json.loads(main_plan.read_text());case_id='poisson2-'+('main' if readout=='matched' else 'equal')+'-s1231'
            update=parent['cases'][case_id]['total_updates'];phase='finish' if readout=='matched' else 'equal'
            checkpoint=a.root/'runs'/('low-depth-main-r2-poisson2-'+phase)/f'checkpoint-u{update}.pt'
            sha=matching[0]['independent_cpu']['checkpoint_sha256']
            if file_sha256(checkpoint)!=sha:raise AssertionError('Context endpoint differs from numerical gate')
            target=out/readout;receipt=target/'report.json';begin=time.time()
            cost_row={'readout':readout,'assigned_chips':0,'started_unix':begin,'completed':False,
                      'reused_completed_score':receipt.exists()}
            costs.append(cost_row)
            if not receipt.exists():
                if target.exists():raise RuntimeError('Preserve partial context score; explicit recovery identity required')
                env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12')
                args=[os.sys.executable,'-u','-m','scripts.evaluate_natural_code_context_recovered','--root',str(a.root),
                    '--checkpoint',str(checkpoint),'--checkpoint-sha',sha,'--plan',str(main_plan),
                    '--plan-sha',file_sha256(main_plan),'--gradient-receipts',
                    'results/parcae_low_depth_pilot_20261024/main-recovery-v2/gradient-progress.json',
                    '--readout',readout,'--output',str(target)]
                with (out/(readout+f'-{int(begin)}.log')).open('x') as log:
                    child=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write_state(public/'status.json',{'status':'SCORING_FROZEN_FOUR_CONTEXT_EXIT_CONDITIONS','readout':readout,'assigned_chips':0})
                while child.poll() is None:
                    guard()
                    if time.time()>begin+plan['seconds_limit_per_endpoint']+30:raise TimeoutError('Context endpoint CPU deadline')
                    time.sleep(5)
                code=child.returncode;cost_row['exit_code']=code;child=None
                if code:raise RuntimeError('Context endpoint independent readout failed')
            row=json.loads(receipt.read_text())
            if row['status']!='COMPLETE_FROZEN_NATURAL_CODE_CONTEXT_READOUT' or row['checkpoint_sha256']!=sha:raise AssertionError('Context readout identity failed')
            write_state(public/(readout+'.json'),row);records.append(row)
            cost_row.update(wall_seconds=time.time()-begin,finished_unix=time.time(),completed=True,exit_code=0)
        result={'status':'COMPLETE_FROZEN_CODE_CONTEXT_READOUTS','records':records}
    except BaseException as error:result.update(error=str(error),records=records);raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        if cost_row and not cost_row['completed']:
            cost_row.update(wall_seconds=time.time()-cost_row['started_unix'],finished_unix=time.time(),
                            exit_code=child.returncode if child else cost_row.get('exit_code'))
        result.update(controller_elapsed_seconds=time.time()-started,cpu_scores=costs,
            test_accessed=False,new_terminal_attempt=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0})


if __name__=='__main__':main()
