"""Sanitized incremental accounting for this mechanism campaign only."""
import argparse,json
from pathlib import Path
from scripts.summarize_recurrent_costs import union_seconds


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 prefixes=('unroll-v1-','untied-','independent-untied-','continuation-v2-','stable-cont-v2-','post-v1-','confirmation-v3-','fast-','fused-amp-')
 rows=[];intervals=[];missing=[];starts=[];ends=[]
 for lp in sorted((a.root/'snapshots').glob('*/launch.json')):
  name=lp.parent.name
  if not name.startswith(prefixes):continue
  launch=json.loads(lp.read_text());cp=lp.with_name('closure.json')
  if not cp.exists():missing.append(name);continue
  closure=json.loads(cp.read_text());report=a.root/'runs'/name/'report.json'
  v=json.loads(report.read_text()) if report.exists() else {};chips=len(launch['physical_chips']);seconds=closure['wall_seconds']
  starts.append(launch['started_unix']);ends.append(launch['started_unix']+seconds)
  if chips:intervals.append((launch['started_unix'],launch['started_unix']+seconds))
  family='unroll' if name.startswith('unroll') else 'untied' if name.startswith(('untied','independent-untied')) else 'longer' if name.startswith(('continuation','stable-cont')) else 'software_performance' if name.startswith(('fast-','fused-amp-')) else 'confirmation'
  rows.append({'run':name,'family':family,'exit_code':closure['exit_code'],'report_status':v.get('status'),
   'assigned_chips':chips,'wall_seconds':seconds,'assigned_chip_hours':chips*seconds/3600,
   'updates':v.get('updates',0),'phase_real_targets':v.get('phase_real_targets',0),
   'warm_real_targets_per_second':v.get('warm_global_real_targets_per_second'),
   'parameters':v.get('parameters'),'scored_real_targets':v.get('targets',0)})
 cpu=[]
 for folder in ['mechanism-states-v1','document-boundary-v1','mechanism-endpoint-cpu-v1','untied-specialization-v1','core-difference-rank-v1','unroll-development-v1','confirmation-campaign-v3/independent-cpu']:
  path=a.root/folder/'report.json'
  if path.exists():
   v=json.loads(path.read_text());cpu.append({'job':folder,'status':v['status'],'wall_seconds':v.get('wall_seconds',v.get('elapsed_seconds',v.get('wall_seconds_including_waits'))),
    'active_scoring_seconds':v.get('active_scoring_seconds'),'configured_threads':16,'accelerator_chips':0})
 span=max(ends)-min(starts) if starts else 0.
 result={'status':'COMPLETE_INCREMENTAL_RECEIPTS' if not missing else 'PARTIAL_INCREMENTAL_RECEIPTS',
  'runs':rows,'missing_closures':missing,'assigned_chip_hours':sum(r['assigned_chip_hours'] for r in rows),
  'accelerator_active_union_hours':union_seconds(intervals)/3600,
  'active_instance_opportunity_chip_hours':union_seconds(intervals)*16/3600,'cpu_only_jobs':cpu,
  'observed_launch_span_hours':span/3600,'observed_span_opportunity_chip_hours':span*16/3600,
  'hours_between_accelerator_launches_in_span':max(0,span-union_seconds(intervals))/3600,
  'scope':'New mechanism campaign and declared follow-up. All closed attempts retained. Native launch time includes compile/setup and recovery. Accelerator-active intervals are unioned before charging the whole node. The separate observed launch span includes CPU work and idle/control gaps between first and last recorded jobs; it is a reservation-cost lower bound, not the full instance bill. CPU work overlaps and is not added again. Reused controls and prior campaigns are historical costs, not charged again. Configured CPU threads are not measured CPU core-hours.'}
 (a.output/'execution-costs.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['runs','cpu_only_jobs']},indent=2))


if __name__=='__main__':main()
