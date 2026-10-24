"""Read-only live progress and honest receipts for the finite mature study."""
import argparse
import json
import time
from pathlib import Path

from compute_trainium.queue import write_state


def collect(root,campaign,output):
    now=time.time();plan=json.loads((campaign/'plan.json').read_text());closed=[];active=[];updates=0;work=0
    for snapshot in sorted((root/'snapshots').glob('mature-fixed-v1-*')):
        launch=json.loads((snapshot/'launch.json').read_text());closure=snapshot/'closure.json'
        name=snapshot.name;report=root/'runs'/name/'report.json'
        if closure.exists():
            row=json.loads(closure.read_text());closed.append({'identity':name,**row})
            if row['exit_code']==0 and report.exists():
                r=json.loads(report.read_text())
                if 'flops_per_global_step' in r:
                    updates+=r['updates'];work+=r['updates']*r['flops_per_global_step']['training_matmul_flops']
        else:
            log=root/'logs'/(name+'.log');rows=[]
            if log.exists():
                for line in log.read_text(errors='replace').splitlines():
                    try:r=json.loads(line)
                    except ValueError:continue
                    if 'update' in r and 'elapsed_seconds' in r and 'rank0_scaled_nll' in r:rows.append(r)
            item={'identity':name,'assigned_physical_chips':len(launch['physical_chips']),
                  'observed_launch_wall_seconds':now-launch['started_unix'],'last_logged_update':rows[-1] if rows else None}
            if len(rows)>=22:
                first,last=rows[-21],rows[-1]
                item['observed_200_update_targets_per_second']=(last['real_tokens']-first['real_tokens'])/(last['elapsed_seconds']-first['elapsed_seconds'])
            active.append(item)
    receipt={'status':'CLOSED_CAMPAIGN_RECEIPTS' if (campaign/'report.json').exists() else 'LIVE_CAMPAIGN_RECEIPTS',
             'collected_unix':now,'closed_attempts':closed,'open_attempts':active,
             'closed_assigned_chip_hours':sum(x['assigned_chip_hours'] for x in closed),
             'open_attempt_opportunity_chip_hours_observed':sum(x['assigned_physical_chips']*x['observed_launch_wall_seconds']/3600 for x in active),
             'declared_window_through_collection_opportunity_chip_hours':16*(now-plan['registered_unix'])/3600,
             'closed_successful_training_updates':updates,'closed_successful_training_matmul_flops':work,
             'scope':'Launch-to-closure costs include setup/compile/checkpoint overhead and failed attempts. Open durations are observations, not successful results. Rolling200-update rates include any checkpoint/timing stalls, exclude earlier setup, and are not a new matched performance qualification. Rank-zero logged NLL is a training diagnostic, not held-out or globally aggregated loss. CPU preparation, prior studies/qualification and parent-copy costs are separate; overlapping intervals must not be added as sequential node time. This is not an entire-instance billing receipt.'}
    if (campaign/'report.json').exists():receipt['campaign_result_status']=json.loads((campaign/'report.json').read_text())['status']
    cpu=campaign/'independent-cpu/report.json';progress=campaign/'independent-cpu/progress.json'
    if cpu.exists():
        d=json.loads(cpu.read_text());receipt['independent_cpu']={k:d[k] for k in ['status','active_seconds','wall_seconds_including_waits']}
    elif progress.exists():
        d=json.loads(progress.read_text());receipt['independent_cpu']={'status':'IN_PROGRESS','completed_endpoint_audits':len(d['records']),
            'active_seconds':sum(x['active_seconds'] for x in d['records'])}
    else:receipt['independent_cpu']={'status':'WAITING_REGISTERED_ENDPOINTS','completed_endpoint_audits':0}
    write_state(output/'execution-costs.json',receipt)
    if (campaign/'state.json').exists():
        state=json.loads((campaign/'state.json').read_text())
        write_state(output/'live-state.json',{'status':state['status'],'heartbeat_unix':state['heartbeat_unix'],
            'stages':{k:{field:v[field] for field in ['status','half','started_unix','ended_unix','result'] if field in v}
                      for k,v in state['stages'].items()}})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--follow',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=True)
    plan=json.loads((a.campaign/'plan.json').read_text())
    while True:
        collect(a.root,a.campaign,a.output)
        if not a.follow or (a.campaign/'report.json').exists() or time.time()>plan['deadline_unix']+300:return
        time.sleep(30)


if __name__=='__main__':main()
