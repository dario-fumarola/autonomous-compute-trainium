"""Complete the predeclared same-weight unroll matrix; no model updates."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(os.environ['RESEARCH_ROOT']);old=root.parent;croot=os.environ['RESEARCH_CONTAINER_ROOT']
    args.output.mkdir(exist_ok=False);started=time.time()
    matrix=[('ordinary',1,1),('recurrent',2,2),('ordinary',1,2),('recurrent',2,1),('recurrent',2,4)]
    spec={'identity':'unroll-diagnostic-v1','seeds':[23,47,71],'matrix':matrix,'split':'dev','cohort':'curve',
          'primary':'Within-checkpoint loss difference, recurrent trained path two passes versus one',
          'scope':'Exploratory mechanism diagnostic, not a replacement for the primary result; no TEST or model selection',
          'evaluator_sha256':file_sha256('scripts/evaluate_recurrent_path.py'),'frozen_unix':time.time()}
    (args.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n');records=[]
    subprocess.run(['docker','update','--memory',str(192*1024**3),'autonomous-compute-neuron233'],check=True,capture_output=True)
    for arm,trained,passes in matrix:
        for seed in spec['seeds']:
            if file_sha256('scripts/evaluate_recurrent_path.py')!=spec['evaluator_sha256']:raise ValueError('Frozen evaluator changed')
            run=f'unroll-v1-{arm}-s{seed}-r{passes}'
            parent=(f'/work/runs/natural-curve-v1-ordinary-s{seed}-u3840' if arm=='ordinary'
                    else f'{croot}/runs/preln-recurrent-v1-s{seed}-u3840')
            a=['--checkpoint',parent+'/checkpoint.pt','--corpus','/work/natural-v2-1050m/dev.json',
               '--repetitions',str(passes),'--cohort','curve']
            if passes==trained:a+=['--calibration-report',parent+'-dev/report.json']
            env=dict(os.environ,RESEARCH_RUN=run,RESEARCH_ENTRY_MODULE='scripts.evaluate_recurrent_path',
                     RESEARCH_DDP_RANKS='64',NEURON_RT_VISIBLE_CORES=','.join(map(str,range(64))),
                     RESEARCH_TIMEOUT_SECONDS='900')
            subprocess.run(['python3','-m','scripts.supervise',*a],env=env,check=True)
            report=json.loads((root/'runs'/run/'report.json').read_text())
            if report['status']!='PASS_SHARDED_RECURRENT_READOUT':raise ValueError('Incomplete readout')
            records.append({'arm':arm,'seed':seed,'trained_passes':trained,'inference_passes':passes,'run':run,**report})
            (args.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
            print(json.dumps({'event':'unroll','arm':arm,'seed':seed,'passes':passes,'nll':report['metrics']['weighted_nll']}),flush=True)
    (args.output/'report.json').write_text(json.dumps({'status':'COMPLETE_SAME_WEIGHT_UNROLL_MATRIX',
        'specification':spec,'records':records,'elapsed_seconds':time.time()-started},indent=2)+'\n')


if __name__=='__main__':main()
