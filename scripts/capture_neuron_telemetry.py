"""Read-only runtime occupancy sampling, with per-application zeros deduplicated."""
import argparse
import json
import select
import statistics
import subprocess
import time
from pathlib import Path


def aggregate(report):
    per_core={};active_processes=0;errors=0
    runtimes=report.get('neuron_runtime_data',[])
    for runtime in runtimes:
        metrics=runtime.get('report',{})
        values=metrics.get('neuroncore_counters',{}).get('neuroncores_in_use',{})
        active=False
        for key,value in values.items():
            utilization=float(value.get('neuroncore_utilization',0))
            per_core[key]=per_core.get(key,0)+utilization
            active=active or utilization>.01
        active_processes+=int(active)
        errors+=sum(metrics.get('execution_stats',{}).get('error_summary',{}).values())
    # Application records include zero entries for all the other cores. Mean
    # over process×core entries would understate occupancy by world size.
    if any(v>101 for v in per_core.values()):raise ValueError('Counter aggregation exceeds physical occupancy; inspect runtime ownership')
    return {'runtime_processes':len(runtimes),'active_runtime_processes':active_processes,
            'reported_logical_cores':len(per_core),'active_logical_cores':sum(v>.01 for v in per_core.values()),
            'mean_reported_core_busy_percent':statistics.mean(per_core.values()) if per_core else None,
            'minimum_reported_core_busy_percent':min(per_core.values()) if per_core else None,
            'maximum_reported_core_busy_percent':max(per_core.values()) if per_core else None,
            'execution_errors':errors}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=12);p.add_argument('--label',required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    config=args.output/'config.json';config.write_text(json.dumps({'period':'1s','neuron_runtimes':[
        {'tag_filter':'.*','metrics':[{'type':'neuroncore_counters'},{'type':'execution_stats'}]}]}))
    proc=subprocess.Popen(['/opt/aws/neuron/bin/neuron-monitor','-c',str(config)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    started=time.monotonic();records=[]
    try:
        with (args.output/'private-raw.jsonl').open('w') as raw:
            while time.monotonic()-started<args.seconds:
                ready,_,_=select.select([proc.stdout],[],[],1)
                if not ready:continue
                line=proc.stdout.readline()
                if not line:break
                try:report=json.loads(line)
                except json.JSONDecodeError:continue
                raw.write(line);raw.flush()
                records.append({'seconds':time.monotonic()-started,**aggregate(report)})
    finally:proc.terminate();proc.wait(timeout=5)
    summary={'label':args.label,'samples':records,
        'scope':'Application occupancy summed by logical core. This is a short runtime-busy sample, not peak-FLOP utilization or a campaign average. Raw PID/socket/instance data remain private.'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
