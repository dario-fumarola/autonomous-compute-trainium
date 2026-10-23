"""Read existing global pre-clip norms; never treat rank-zero NLL as DEV."""
import argparse,json,statistics
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--old-root',type=Path,required=True);p.add_argument('--new-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows=[]
 for arm in ['ordinary','recurrent']:
  root=a.old_root if arm=='ordinary' else a.new_root
  for seed in [23,47,71]:
   history={}
   for u in ([384,1152,3840] if arm=='ordinary' else [384,1152,2042,3456,3840]):
    run=(f'natural-curve-v1-ordinary-s{seed}-u{u}' if arm=='ordinary' else f'preln-recurrent-v1-s{seed}-u{u}')
    report=json.loads((root/'runs'/run/'report.json').read_text())
    for r in report['progress']:
     if r['update'] in history and history[r['update']]!=r:raise ValueError('Duplicated update with conflicting history')
     history[r['update']]=r
   for low,high,label in [(0,40,'warmup'),(41,384,'early'),(385,1152,'middle'),(1153,3456,'later_stable'),(3457,3840,'decay')]:
    values=[r['gradient_norm'] for u,r in history.items() if low<=u<=high]
    if not values:continue
    ordered=sorted(values)
    rows.append({'arm':arm,'seed':seed,'period':label,'first_update':low,'last_update':high,'logged_samples':len(values),
     'mean_norm':statistics.mean(values),'median_norm':statistics.median(values),'maximum_norm':max(values),
     'p95_norm':ordered[round(.95*(len(ordered)-1))],'samples_above_clip_1':sum(x>1 for x in values),
     'fraction_logged_samples_above_clip_1':sum(x>1 for x in values)/len(values)})
 result={'status':'COMPLETE_LOGGED_GRADIENT_NORM_DIAGNOSTIC','rows':rows,
  'scope':'Global DDP-averaged gradient norm BEFORE clipping. Approximately every ten updates plus phase boundaries. Sampling is not a complete clipping-frequency trace; rank-zero NLL is not analyzed as a generalization gap. Branch-only compute endpoints excluded.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)


if __name__=='__main__':main()
