"""UNEXECUTED HARDWARE DRAFT — not a qualified research trainer.

Outcome-independent measured allocation for the predeclared 3B readout."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from compute_trainium.training import file_sha256


def allocate(seconds, targets_per_seed, rates, overhead_per_pair=650., safety=1.25):
    cost=targets_per_seed*sum(1/rates[str(r)] for r in [1,2])*safety/2+overhead_per_pair
    # Paired jobs finish at the slower recurrent job: use that bound, not an
    # optimistic arithmetic mean when the ordinary half goes idle.
    cost=max(cost,targets_per_seed/min(rates.values())*safety+overhead_per_pair)
    return cost<=seconds,cost


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--primary',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True)
    args=p.parse_args();root=Path(os.environ['RESEARCH_ROOT']);old=root.parent;croot=os.environ['RESEARCH_CONTAINER_ROOT']
    if not (args.primary/'complete.json').exists():raise ValueError('Primary must finish first')
    q=json.loads(args.qualification.read_text())
    if q['status']!='PASS_FAST_PATH_QUALIFICATION':raise ValueError('Fast path is not qualified')
    if not q['full_optimizer_recovery_both_arms']:raise ValueError('Full recovery missing')
    manifest=json.loads((args.data/'manifest.json').read_text())
    if manifest['status']!='PASS_FROZEN_NATURAL_EXTENSION':raise ValueError('Append-only source gate failed')
    args.output.mkdir(exist_ok=False);prefix=np.load(args.data/'packed/prefix_counts.npy',mmap_mode='r')
    targets=lambda u:int(prefix[u*256].sum())
    remaining=args.deadline_unix-time.time()-900 # Protect final analysis, backups and push.
    rates={k:float(v) for k,v in q['rates'].items()}
    three,cost3=allocate(remaining,3*(targets(11520)-targets(1152)),rates,overhead_per_pair=3*650)
    single,cost1=allocate(remaining,targets(11520)-targets(3456),rates)
    seeds=[23,47,71] if three else ([23] if single else [])
    fork=1152 if three else 3456
    allocation={'status':'FROZEN_LONGER_ALLOCATION','selected_seeds':seeds,'fork_update':fork,
        'endpoint_update':11520,'decay_start_update':10368,'remaining_seconds_after_final_reserve':remaining,
        'three_pair_conservative_seconds':cost3,'one_pair_conservative_seconds':cost1,'measured_rates':rates,
        'criterion':'Data gate and qualified measured time only; no gain or seed-outcome selection',
        'deadline_unix':args.deadline_unix,'frozen_unix':time.time(),
        'qualification_sha256':file_sha256(args.qualification),'data_manifest_sha256':file_sha256(args.data/'manifest.json'),
        'code_sha256':{f:file_sha256(f) for f in ['scripts/train_fast.py','scripts/migrate_checkpoint.py',
            'compute_trainium/checkpoint_migration.py','scripts/evaluate_recurrent_path.py',
            'scripts/run_longer_recurrence.py']}}
    (args.output/'allocation.json').write_text(json.dumps(allocation,indent=2)+'\n')
    if not seeds:
        (args.output/'not-launched.json').write_text(json.dumps(allocation,indent=2)+'\n');return
    records=[];started=time.time()
    def execute(name,module,a,cores,timeout=7200):
        for f,sha in allocation['code_sha256'].items():
            if file_sha256(f)!=sha:raise ValueError('Frozen longer source changed')
        if time.time()+300>args.deadline_unix:raise TimeoutError('No new phase near deadline')
        env=dict(os.environ,RESEARCH_RUN=name,RESEARCH_ENTRY_MODULE=module,RESEARCH_DDP_RANKS='32',
                 NEURON_RT_VISIBLE_CORES=','.join(map(str,cores)),RESEARCH_TIMEOUT_SECONDS=str(timeout))
        subprocess.run(['python3','-m','scripts.supervise',*a],env=env,check=True)
        return json.loads((root/'runs'/name/'report.json').read_text())
    def one(seed,r):
        arm='ordinary' if r==1 else 'recurrent';base=f'long3b-{arm}-s{seed}'
        parent=(old/f'runs/natural-curve-v1-ordinary-s{seed}-u{fork}/checkpoint.pt' if r==1
                else root/f'runs/preln-recurrent-v1-s{seed}-u{fork}/checkpoint.pt')
        if fork==3456 and r==1:parent=old/f'runs/natural-curve-v1-ordinary-s{seed}-u3840/stable-u3456.pt'
        migrated=root/'migrations'/(base+'.pt');migrated.parent.mkdir(exist_ok=True)
        subprocess.run([os.sys.executable,'-m','scripts.migrate_checkpoint','--checkpoint',str(parent),'--data',str(args.data),
            '--output',str(migrated),'--optimizer','fused','--decay-start-update','10368','--end-update','11520'],check=True)
        cores=list(range((r-1)*32,r*32));cp=f'{croot}/migrations/{base}.pt';previous=fork
        common=['--size','small','--architecture','ordinary' if r==1 else 'preln','--repetitions',str(r),
            '--context','1024','--batch','4','--accumulation','2','--precision','bf16','--optimizer','fused',
            '--data-seed','23','--seed',str(seed),'--reference-threads','4','--purpose','scientific',
            '--corpus',f'{croot}/natural-append-3250m/source.json',
            '--packed-corpus',f'{croot}/natural-append-3250m/packed','--dev-corpus','/work/natural-v2-1050m/dev.json',
            '--lr','.001' if r==1 else '.0003','--lr-warmup-tokens',str(targets(40)),
            '--decay-start',str(targets(10368)),'--decay-end',str(targets(11520)),
            '--checkpoint-every','384','--warmup','4','--pauses','0','--save']
        output=[]
        for u in [10368,11520]:
            name=f'{base}-u{u}'
            train=execute(name,'scripts.train_fast',[*common,'--resume',cp,'--steps',str(u-previous)],cores)
            if train['total_real_targets']!=targets(u) or not train['all_replica_model_optimizer_states_equal']:
                raise AssertionError('Extended exposure or full replica state differs')
            cp=f'{croot}/runs/{name}/checkpoint.pt';previous=u
            ev=execute(name+'-dev','scripts.evaluate_recurrent_path',[
                '--corpus','/work/natural-v2-1050m/dev.json','--checkpoint',cp,'--repetitions',str(r)],cores,900)
            row={'run':name,'seed':seed,'arm':arm,'update':u,'real_tokens':train['total_real_targets'],
                'metrics':ev['metrics'],'checkpoint':cp,'initial_model_digest':train['initial_model_digest'],
                'training_matmul_flops':train['total_training_matmul_flops'],'state_digest':train['state_digest']}
            output.append(row)
            (args.output/(base+'-progress.json')).write_text(json.dumps(output,indent=2)+'\n')
            print(json.dumps({'event':'longer_curve',**{k:row[k] for k in ['seed','arm','update','real_tokens']},'nll':ev['metrics']['weighted_nll']}),flush=True)
        return output
    try:
        for seed in seeds:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(one,seed,r) for r in [1,2]]
                for f in futures:records.extend(f.result())
            (args.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        (args.output/'complete.json').write_text(json.dumps({'status':'COMPLETE_LONGER_RECURRENT_COMPARISON',
            'allocation':allocation,'records':records,'elapsed_seconds':time.time()-started},indent=2)+'\n')
    except BaseException as e:
        (args.output/'failure.json').write_text(json.dumps({'error':str(e),'type':type(e).__name__,'elapsed_seconds':time.time()-started},indent=2)+'\n');raise


if __name__=='__main__':main()
