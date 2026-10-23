"""Publish content-free progress from actual completed receipts and live logs."""
import argparse
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    state=json.loads((a.campaign/'state.json').read_text());active=[]
    for name,stage in state['stages'].items():
        if stage.get('status')!='RUNNING' or not name.startswith('vdepth-'):continue
        record={'identity':name,'assigned_chips':8,'module':stage['module']}
        log=a.root/'logs'/(name+'.log');last=None
        if log.exists():
            for line in log.read_text().splitlines():
                if not line.startswith('{'):continue
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if 'update' in row and 'real_tokens' in row and 'rank0_scaled_nll' in row:last=row
        if last:record.update({k:last[k] for k in ['update','real_tokens','lr','gradient_norm','elapsed_seconds']})
        # Rank-zero loss is a shard diagnostic, never a global evaluation score.
        active.append(record)
    tuning=[]
    for path in sorted((a.root/'runs').glob('vdepth-v1-tune-*-dev-r*/report.json')):
        report=json.loads(path.read_text());tuning.append({'identity':path.parent.name,'cohort':report['cohort'],
             'weighted_nll':report['metrics']['weighted_nll'],'targets':report['targets']})
    curves=[]
    for path in sorted((a.root/'runs').glob('vdepth-v1-*-s*-u*-dev-r*/report.json')):
        report=json.loads(path.read_text());curves.append({'identity':path.parent.name,'training_real_targets':report['training_real_tokens'],
             'depth':report['inference_repetitions'],'metrics':report['metrics']})
    completed=(a.campaign/'report.json').exists()
    result={'updated_unix':time.time(),'status':json.loads((a.campaign/'report.json').read_text())['status'] if completed else 'RUNNING',
       'active_phases':active,'completed_tuning_trials':tuning,'completed_learning_curve_cells':curves,
       'stage_states':{k:v['status'] for k,v in state['stages'].items() if not k.startswith('vdepth-')},
       'scope':'Only actual native progress and completed readouts. Training rank-zero loss is omitted; TUNE selects the declared LR only. Stable and annealed curve endpoints differ. No best-depth or TEST selection.'}
    a.output.mkdir(parents=True,exist_ok=True)
    temporary=a.output/'progress.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(a.output/'progress.json')
    print(json.dumps({'status':result['status'],'active':len(active),'completed_tuning':len(tuning),'curve_cells':len(curves)}))


if __name__=='__main__':main()
