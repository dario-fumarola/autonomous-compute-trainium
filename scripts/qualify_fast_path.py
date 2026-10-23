"""Bounded qualification of fused AdamW and concurrent half-node jobs.

Executed evidence is in results/fast_path_20261023. Software qualification
does not select scientific models or authorize another training campaign.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);args=p.parse_args()
    root=Path(os.environ['RESEARCH_ROOT']);croot=os.environ['RESEARCH_CONTAINER_ROOT']
    args.output.mkdir(exist_ok=False);started=time.time();events=[]
    subprocess.run(['docker','update','--memory',str(352*1024**3),'autonomous-compute-neuron233'],check=True,capture_output=True)
    def phase(name,module,arguments,cores):
        env=dict(os.environ,RESEARCH_RUN=name,RESEARCH_ENTRY_MODULE=module,
                 RESEARCH_DDP_RANKS=str(len(cores)),NEURON_RT_VISIBLE_CORES=','.join(map(str,cores)),
                 RESEARCH_TIMEOUT_SECONDS='900')
        log=(args.output/(name+'.log')).open('w')
        return subprocess.Popen(['python3','-m','scripts.supervise',*arguments],env=env,stdout=log,stderr=subprocess.STDOUT),log
    def finish(jobs):
        failed=False
        for process,log in jobs:
            code=process.wait();log.close();failed|=code!=0
        if failed:raise RuntimeError('Fast-path qualification failed; preserve receipts and stop')
        if time.time()>args.deadline_unix:raise TimeoutError('Qualification deadline')
    finish([phase('fast-adamw-fixture-001','scripts.qualify_fused_adamw',['--mode','fused'],[0])])
    fixture=json.loads((root/'runs/fast-adamw-fixture-001/report.json').read_text())
    if fixture['status']!='PASS_OPTIMIZER_SOFTWARE_FIXTURE':raise ValueError('Fixture failed')
    base=['--corpus','/work/natural-v2-1050m/source.json','--packed-corpus','/work/natural-v2-1050m/packed',
          '--dev-corpus','/work/natural-v2-1050m/dev.json','--context','1024','--batch','4','--accumulation','2',
          '--precision','bf16','--data-seed','23','--reference-threads','4','--optimizer','fused',
          '--pauses','0','--warmup','4','--seed','23','--save','--architecture','preln']
    finish([phase('fast-tiny-global32-001','scripts.train_fast',base+['--size','tiny','--repetitions','2','--steps','8','--check-global'],list(range(32)))])
    tiny=json.loads((root/'runs/fast-tiny-global32-001/report.json').read_text())
    if not tiny['global_gradient_check'] or not tiny['all_replica_model_optimizer_states_equal']:
        raise ValueError('Global-gradient/replica proof missing')
    jobs=[]
    peers=','.join(f'fast-150m-r{r}-half32-001' for r in [1,2])
    rendezvous=f'{croot}/{args.output.relative_to(root)}/warm-ready'
    for r in [1,2]:
        jobs.append(phase(f'fast-150m-r{r}-half32-001','scripts.train_fast',base+['--size','small','--repetitions',str(r),'--steps','104',
            '--warm-rendezvous',rendezvous,'--warm-peer-names',peers],list(range((r-1)*32,r*32))))
    finish(jobs)
    reports={r:json.loads((root/f'runs/fast-150m-r{r}-half32-001/report.json').read_text()) for r in [1,2]}
    overlap=max(0,min(v['warm_ended_unix'] for v in reports.values())-max(v['warm_started_unix'] for v in reports.values()))
    if overlap<.95*min(v['warm_seconds'] for v in reports.values()):
        raise AssertionError('Two warm timing windows did not overlap sufficiently')
    # Independent process recovery of both full model + optimizer states.
    jobs=[]
    for r in [1,2]:
        jobs.append(phase(f'fast-restore-r{r}-half32-001','scripts.train_fast',base+['--size','small','--repetitions',str(r),'--steps','8',
            '--resume',f'{croot}/runs/fast-150m-r{r}-half32-001/checkpoint.pt'],list(range((r-1)*32,r*32))))
    finish(jobs)
    for r in [1,2]:
        recovered=json.loads((root/f'runs/fast-restore-r{r}-half32-001/report.json').read_text())
        if recovered['recovered_model_optimizer_digest']!=reports[r]['state_digest']:
            raise AssertionError('Full half-node checkpoint recovery differs')
        if not recovered['all_replica_model_optimizer_states_equal']:raise AssertionError('Replicas differ')
    result={'status':'PASS_FAST_PATH_QUALIFICATION','elapsed_seconds':time.time()-started,
        'optimizer':'fused','world_per_job':32,'batch_per_rank':4,'accumulation':2,
        'global_microbatch_windows':128,'global_update_windows':256,'warm_updates_per_arm':100,
        'concurrent_half_node_measurements':True,'fp32_master_and_moments':True,
        'synchronized_warm_start':True,'warm_overlap_seconds':overlap,
        'warm_overlap_fraction_by_arm':{str(r):overlap/reports[r]['warm_seconds'] for r in [1,2]},
        'optimizer_fixture':fixture,'tiny_global_gradient':tiny['global_gradient_check'],
        'rates':{str(r):reports[r]['warm_global_real_targets_per_second'] for r in [1,2]},
        'full_optimizer_recovery_both_arms':True,'replicas_identical_both_arms':True,
        'main_reports':{str(r):reports[r] for r in [1,2]}}
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='main_reports'}),flush=True)


if __name__=='__main__':main()
