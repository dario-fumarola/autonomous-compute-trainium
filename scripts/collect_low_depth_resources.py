"""Observed chip-time ledger; overlap and partial scientific work stay explicit."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def union_seconds(intervals):
    total=0.;end=None
    for start,stop in sorted(intervals):
        if stop<start:raise ValueError('Negative measured execution interval')
        if end is None or start>end:total+=stop-start;end=stop
        elif stop>end:total+=stop-end;end=stop
    return total


def scientific_work(root,identity,report,cutoff):
    if not identity.startswith(('low-depth-tune-','low-depth-main-v1-')):return None
    if report and report.get('status')=='COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT':
        return {'status':'COMPLETE_EXACT_SEGMENT_COUNTER','updates':report['actual_optimizer_calls_per_replica'],
                'real_target_presentations':report['segment_real_targets'],
                'matrix_operations':report['segment_executed_matmuls']}
    # Only the preserved tuning-v1 stop has an interrupted scientific phase so
    # far. Future incomplete identities remain explicitly uncounted,not zero.
    if not identity.startswith('low-depth-tune-v1-'):return {'status':'INCOMPLETE_WORK_NOT_EXACTLY_COUNTED','matrix_operations':None}
    log=root/'logs'/(identity+'.log')
    rows=[json.loads(s) for s in log.read_text().splitlines() if s.startswith('{"update":')] if log.exists() else []
    if not rows:return {'status':'NO_COMPLETED_UPDATE_LOG_AVAILABLE','matrix_operations':None}
    arm=next(a for a in ['ordinary','fixed1','poisson2'] if '-'+a+'-' in identity)
    experiment=root/'parcae-low-depth-pilot-v1';plan=json.loads((experiment/'tuning-plan.json').read_text())
    case=next(c for c in plan['cases'].values() if c['arm']==arm)
    work=np.load(experiment/case['work_table'],mmap_mode='r')
    if file_sha256(experiment/case['work_table'])!=case['work_table_sha256']:raise AssertionError('Interrupted-work bound table changed')
    origin=20 if identity.endswith('-fit') else 0;last=rows[-1]['update']
    upper=min(last+11,case['total_updates']) # Up to10 completed unlogged updates plus a partial next update.
    return {'status':'BOUNDED_INTERRUPTED_WORK_FROM_IMMUTABLE_SCHEDULE_AND_LOGS',
        'logged_completed_update':last,'segment_origin_update':origin,
        'completed_optimizer_updates_lower_bound':max(0,last-origin),
        'completed_optimizer_updates_upper_bound':max(0,min(last+10,case['total_updates'])-origin),
        'matrix_operations_lower_bound':int(work[origin:last].sum()),
        'matrix_operations_upper_bound':int(work[origin:upper].sum()),
        'scope':'Progress logs occur after every10th update. Upper work also allows a partly executed next objective. No claim of exact unseen work after termination.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();cutoff=time.time();jobs=[]
    prefixes=('parcae-','low-depth-')
    for launch_file in sorted((a.root/'snapshots').glob('*/launch.json')):
        identity=launch_file.parent.name;launch=json.loads(launch_file.read_text());start=launch['started_unix']
        if start>cutoff:continue
        closure_file=launch_file.parent/'closure.json';closure=json.loads(closure_file.read_text()) if closure_file.exists() else None
        finish=min(cutoff,start+closure['wall_seconds']) if closure else cutoff
        report_file=a.root/'runs'/identity/'report.json';report=json.loads(report_file.read_text()) if report_file.exists() else None
        jobs.append({'identity':identity,'started_unix':start,'finished_unix':finish,'chips':launch['physical_chips'],
            'closed':closure is not None,'exit_code':closure['exit_code'] if closure else None,
            'report_status':report.get('status') if report else None,'new_family':identity.startswith(prefixes),
            'scientific_work':scientific_work(a.root,identity,report,cutoff)})
    selected=[r for r in jobs if r['new_family']]
    if not selected:raise ValueError('No measured low-depth family launch intervals')
    start=min(r['started_unix'] for r in selected);window=cutoff-start
    per_chip={chip:[] for chip in range(16)}
    for row in jobs:
        if row['finished_unix']<=start:continue
        for chip in row['chips']:per_chip[chip].append((max(start,row['started_unix']),row['finished_unix']))
    all_assigned=sum(union_seconds(v) for v in per_chip.values())
    family_assigned=sum((r['finished_unix']-r['started_unix'])*len(r['chips']) for r in selected)
    exact=[r['scientific_work'] for r in selected if r['scientific_work'] and r['scientific_work']['status']=='COMPLETE_EXACT_SEGMENT_COUNTER']
    bounded=[r['scientific_work'] for r in selected if r['scientific_work'] and r['scientific_work']['status'].startswith('BOUNDED_')]
    result={'status':'OBSERVED_LOW_DEPTH_RESOURCE_LEDGER','observed_unix':cutoff,
        'window_started_unix':start,'window_seconds':window,'sixteen_chip_opportunity_hours':window*16/3600,
        'new_family_assigned_chip_hours':family_assigned/3600,
        'all_project_native_unique_assigned_chip_hours_within_window':all_assigned/3600,
        'unassigned_native_chip_hours_within_window':max(0,window*16-all_assigned)/3600,
        'new_family_native_attempts':len(selected),'closed_attempts':sum(r['closed'] for r in selected),
        'failed_closed_attempts':sum(r['closed'] and r['exit_code']!=0 for r in selected),
        'exact_completed_scientific_segment_matrix_operations':sum(r['matrix_operations'] for r in exact),
        'exact_completed_scientific_target_presentations':sum(r['real_target_presentations'] for r in exact),
        'additional_interrupted_scientific_matrix_operations_bounds':[
            sum(r['matrix_operations_lower_bound'] for r in bounded),sum(r['matrix_operations_upper_bound'] for r in bounded)],
        'native_attempts':[{**{k:v for k,v in r.items() if k not in ['chips','new_family']},
            'assigned_chips':len(r['chips']),'assigned_chip_hours':(r['finished_unix']-r['started_unix'])*len(r['chips'])/3600} for r in selected],
        'source_sha256':file_sha256(__file__),'test_accessed':False,
        'scope':'All parcae-/low-depth- launch attempts,including failed/stopped native work,each immutable identity once; retained phases are not charged twice. Window also reports other project occupancy without double-counting overlapping chip intervals. Assigned chips are resource reservations,not hardware utilization. Launch intervals omit earlier uninstrumented container bootstrap; whole opportunity window includes gaps/CPU-only waiting. CPU preparation/audit/eval task costs remain in their execution receipts. Scientific matrix totals exclude qualification/evaluation matrix work,which is charged in chip time and individual receipts. Target presentations across models/repeated branches are not unique corpus tokens. Matrix-operation matching is not wall-time matching.'}
    write_state(a.output,result)
    print(json.dumps({k:result[k] for k in ['new_family_native_attempts','new_family_assigned_chip_hours','sixteen_chip_opportunity_hours','failed_closed_attempts']}))


if __name__=='__main__':main()
