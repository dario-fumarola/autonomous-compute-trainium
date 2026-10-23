"""All native attempts and known NN work for the separate variable-depth study."""
import argparse
import json
import time
from pathlib import Path


def union_seconds(intervals):
    ends=[]
    for start,end in sorted(intervals):
        if not ends or start>ends[-1][1]:ends.append([start,end])
        else:ends[-1][1]=max(ends[-1][1],end)
    return sum(end-start for start,end in ends)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=[];intervals=[];missing=[];training=probes=readout=0;unknown=[]
    for directory in sorted((a.root/'snapshots').glob('vdepth-*')):
        if not (directory/'launch.json').exists():continue
        launch=json.loads((directory/'launch.json').read_text())
        closure_path=directory/'closure.json'
        if not closure_path.exists():missing.append(directory.name);continue
        closure=json.loads(closure_path.read_text());begin=launch['started_unix'];end=begin+closure['wall_seconds']
        chips=len(launch['physical_chips']);intervals.append((begin,end))
        row={'identity':directory.name,'module':launch.get('entry_module'),'started_unix':begin,
             'wall_seconds':closure['wall_seconds'],'assigned_chips':chips,'assigned_chip_hours':chips*(end-begin)/3600,
             'exit_code':closure['exit_code'],'report_exists':closure['report_exists']}
        report_path=a.root/'runs'/directory.name/'report.json'
        if report_path.exists():
            report=json.loads(report_path.read_text());row['status']=report['status']
            if 'flops_per_global_step' in report:
                per=report['flops_per_global_step']['training_matmul_flops']
                row['known_optimizer_update_matmul_flops']=per*report['updates']
                row['known_numerical_probe_matmul_flops']=per//report['accumulation']
                training+=row['known_optimizer_update_matmul_flops'];probes+=row['known_numerical_probe_matmul_flops']
                row['real_training_targets']=report['phase_real_targets']
            elif 'forward_flops_per_window' in report:
                row['known_readout_matmul_flops']=report['forward_flops_per_window']*(report['scored_windows']+report['native_probe_windows'])
                readout+=row['known_readout_matmul_flops'];row['real_scored_targets']=report['targets']
        else:
            row['status']='FAILED_WITHOUT_NORMAL_RESULT'
            checkpoint=a.root/'runs'/directory.name/'checkpoint.pt'
            if checkpoint.exists() and '--resume' not in launch['arguments']:
                import torch
                saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
                row['known_optimizer_update_matmul_flops']=saved['training_matmul_flops']
                row['known_numerical_probe_matmul_flops']=saved['training_matmul_flops']//saved['update']//saved['signature']['accumulation']
                row['real_training_targets']=saved['real_tokens'];row['work_source']='verified saved checkpoint metadata'
                training+=row['known_optimizer_update_matmul_flops'];probes+=row['known_numerical_probe_matmul_flops'];del saved
            else:unknown.append(directory.name)
        records.append(row)
    active=union_seconds(intervals);span=(max(end for _,end in intervals)-min(begin for begin,_ in intervals)) if intervals else 0
    import datetime as dt
    start=dt.datetime(2026,10,23,20,15,tzinfo=dt.timezone.utc).timestamp()
    last=max((end for _,end in intervals),default=start)
    fresh=json.loads((a.root/'variable-depth-fresh-v1/manifest.json').read_text())
    result={'status':'COMPLETE_VARIABLE_DEPTH_RECEIPTS' if not missing else 'PARTIAL_VARIABLE_DEPTH_RECEIPTS',
      'closed_native_attempts':len(records),'runs':records,'missing_closures':missing,
      'assigned_chip_hours':sum(v['assigned_chip_hours'] for v in records),
      'accelerator_active_union_hours':active/3600,'active_node_opportunity_chip_hours':16*active/3600,
      'observed_native_launch_span_hours':span/3600,'declared_window_through_last_closed_launch_hours':(last-start)/3600,
      'declared_window_through_last_closed_launch_opportunity_chip_hours':16*(last-start)/3600,
      'known_optimizer_update_matmul_flops':training,'known_numerical_probe_matmul_flops':probes,
      'known_readout_matmul_flops':readout,'unquantified_failed_nn_work':unknown,
      'fresh_panel_preparation_seconds':fresh['preparation_seconds'],'fresh_panel_configured_workers':20,
      'scope':'All variable-depth software, failed/stopped attempts, tuning, fits and readouts. Native launch includes setup/compile/full-state checks. Union active intervals avoids overlap double-counting. Declared-window span includes CPU/control/idle time but is not the entire instance bill. NN work includes every completed optimizer update, separately known native numerical probes and readout forwards; excludes CPU math, elementwise/optimizer kernels, communication, compiler-internal and unobserved failed NN work, all retained in wall cost. Reused raw downloads/corpora and prior campaign fits remain historical costs. Configured CPU workers are not measured core-hours.'}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'execution-costs.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','scope']}))


if __name__=='__main__':main()
