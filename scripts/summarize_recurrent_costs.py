"""Aggregate sanitized local execution receipts, including failed attempts."""
import argparse
import datetime as dt
import json
from pathlib import Path


def union_seconds(intervals):
    end=None;total=0.
    for a,b in sorted(intervals):
        if end is None or a>=end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-root',type=Path,required=True)
    p.add_argument('--new-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[];intervals=[];missing=[]
    for root,family in [(args.old_root/'recurrence','initial_screen'),(args.new_root,'preln_primary')]:
        for launch in sorted((root/'snapshots').glob('*/launch.json')):
            identity=launch.parent.name;v=json.loads(launch.read_text());cp=launch.with_name('closure.json')
            if not cp.exists():missing.append({'run':identity,'reason':'missing closure'});continue
            c=json.loads(cp.read_text());chips=len(v['physical_chips']);seconds=c['wall_seconds']
            if chips:intervals.append((v['started_unix'],v['started_unix']+seconds))
            report=root/'runs'/identity/'report.json';r=json.loads(report.read_text()) if report.exists() else {}
            rows.append({'run':identity,'family':family,'exit_code':c['exit_code'],
                'wall_seconds':seconds,'assigned_chips':chips,'assigned_chip_hours':seconds*chips/3600,
                'report_status':r.get('status'),'actual_updates':r.get('updates',0),
                'phase_real_targets':r.get('phase_real_targets',0),
                'warm_targets_per_second':r.get('warm_global_real_targets_per_second')})
    result={'status':'SANITIZED_EXECUTION_COST_LEDGER','runs':rows,'missing_closures':missing,
        'assigned_chip_hours':sum(r['assigned_chip_hours'] for r in rows),
        'active_instance_union_hours':union_seconds(intervals)/3600,
        'active_instance_opportunity_chip_hours':union_seconds(intervals)*16/3600,
        'scope':'This recurrent campaign only. Reused ordinary controls were charged in the preserved prior campaign. All closed native attempts included, not just final fits. CPU preparation and idle reservation time are separate.',
        'families':{f:{'runs':sum(r['family']==f for r in rows),'assigned_chip_hours':sum(r['assigned_chip_hours'] for r in rows if r['family']==f),
            'actual_updates':sum(r['actual_updates'] for r in rows if r['family']==f),
            'real_targets_processed':sum(r['phase_real_targets'] for r in rows if r['family']==f)} for f in ['initial_screen','preln_primary']}}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs']},indent=2))


if __name__=='__main__':main()
