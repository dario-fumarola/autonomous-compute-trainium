"""Bounded, frozen fixed-recurrence campaign on two disjoint eight-chip groups."""
import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from compute_trainium.model import ModelConfig
from compute_trainium.recurrence import RecurrenceConfig
from compute_trainium.training import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--identity', default='recurrent-natural-v1')
    p.add_argument('--deadline-unix', type=float, required=True)
    args = p.parse_args()
    root = Path(os.environ['RESEARCH_ROOT'])
    container_root = os.environ['RESEARCH_CONTAINER_ROOT']
    data_manifest = json.loads((args.data/'manifest.json').read_text())
    if data_manifest['status']!='PASS_FROZEN_NATURAL_CORPUS' or data_manifest['source_group_overlap']:
        raise ValueError('Frozen data gate failed')
    qualification_names = ['recurrence-150m-r1-ddp64-001','recurrence-150m-r2-ddp64-001',
        'recurrence-150m-r2-half0-001','recurrence-150m-r2-half1-001',
        'recurrence-150m-r1-half0-001','recurrence-150m-r2-half1-restore-001']
    qualifications = {}
    for name in qualification_names:
        report = json.loads((root/'runs'/name/'report.json').read_text())
        receipt = json.loads((root/'logs'/(name+'.supervisor.json')).read_text())
        if (report['status']!='PASS_PACKED_TRAINING_PHASE' or report['warm_updates']<100
            or not report['all_replica_model_optimizer_states_equal'] or receipt['worker_exit_code']
            or receipt['supervisor_failure']): raise ValueError('Native gate failed: '+name)
        qualifications[name]=report
    recovered = qualifications['recurrence-150m-r2-half1-restore-001']
    previous = qualifications['recurrence-150m-r2-half1-001']
    if recovered['recovered_model_optimizer_digest'] != previous['state_digest']:
        raise ValueError('Full recovered state does not match prior state')
    for name in ['recurrence-cpu-001','recurrence-tiny-ddp64-001']:
        report=json.loads((root/'runs'/name/'report.json').read_text())
        receipt=json.loads((root/'logs'/(name+'.supervisor.json')).read_text())
        if not report['status'].startswith('PASS') or receipt['worker_exit_code'] or receipt['supervisor_failure']:
            raise ValueError('CPU/global reference gate failed')
    half_rate=sum(qualifications[f'recurrence-150m-r2-half{i}-001']['warm_global_real_targets_per_second'] for i in [0,1])
    full_rate=qualifications['recurrence-150m-r2-ddp64-001']['warm_global_real_targets_per_second']
    if half_rate <= full_rate: raise ValueError('Measured half-node topology did not win')
    args.output.mkdir(exist_ok=False)
    started=time.time()
    prefix=np.load(args.data/'packed/prefix_counts.npy',mmap_mode='r')
    def targets(step): return int(prefix[step*256].sum())
    costs={r:RecurrenceConfig(repetitions=r).flops(1024,1024,256)['training_matmul_flops'] for r in [1,2]}
    q=math.floor(3840*costs[1]/costs[2]); fork=q-math.ceil(.1*q)
    steps=sorted({16,40,346,384,1152,fork,q,3456,3840})
    rates={1:qualifications['recurrence-150m-r1-half0-001']['warm_global_real_targets_per_second'],
           2:min(qualifications[f'recurrence-150m-r2-half{i}-001']['warm_global_real_targets_per_second'] for i in [0,1])}
    warm_seconds=3*targets(3840)/rates[1]+3*(targets(3840)+targets(q)-targets(fork))/rates[2]
    tuning_seconds=3*targets(384)/rates[1]+3*targets(384)/rates[2]
    # Two jobs run concurrently; allow 180 seconds cold/eval/save overhead per
    # training+DEV pair, plus a 50% warm-time contingency.
    estimated_seconds=(1.5*(warm_seconds+tuning_seconds)+180*36)/2
    if time.time()+estimated_seconds > args.deadline_unix:
        raise ValueError('Measured conservative campaign budget exceeds the deadline')
    core_files=['compute_trainium/model.py','compute_trainium/titan.py','compute_trainium/recurrence.py',
                'compute_trainium/data.py','compute_trainium/training.py','compute_trainium/metrics.py',
                'scripts/train_distributed.py','scripts/evaluate_checkpoint.py']
    specification={'identity':args.identity,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'protocol_sha256':file_sha256('docs/RECURRENCE_PROTOCOL.md'),
        'data_manifest_sha256':file_sha256(args.data/'manifest.json'),
        'dev_sha256':file_sha256(args.data/'dev.json'),'sealed_test_sha256':file_sha256(args.data/'test-sealed.json'),
        'code_sha256':{f:file_sha256(f) for f in core_files},'seeds':[23,47,71],
        'tuning_seed':11,'lrs':[3e-4,1e-3,3e-3],'global_windows_per_update':256,
        'world_per_job':32,'concurrent_jobs':2,'batch':2,'accumulation':4,'data_seed':23,
        'arms':{'single':1,'recurrent':2},'paired_parameters':RecurrenceConfig().parameter_count(),
        'boundaries_real_targets':{str(s):targets(s) for s in steps},
        'flops_per_global_update':costs,'compute_endpoint_update':q,'compute_fork_update':fork,
        'compute_relative_rounding_deficit':1-q*costs[2]/(3840*costs[1]),
        'selected_topology':'Two disjoint 32-rank / eight-chip jobs; constant 256 windows/update',
        'full_16_chip_recurrent_targets_per_second':full_rate,
        'concurrent_halves_recurrent_targets_per_second':half_rate,
        'topology_aggregate_speedup':half_rate/full_rate,
        'budget':{'conservative_remaining_seconds':estimated_seconds,'deadline_unix':args.deadline_unix,
                  'warm_half_job_rates':rates,'shared_container_memory_gib':288},
        'primary_cohort':'curve','test_sealed':True,'claim_threshold_nats':.01}
    (args.output/'specification.json').write_text(json.dumps(specification,indent=2)+'\n')
    common=['--size','small','--architecture','recurrent','--context','1024','--batch','2',
            '--accumulation','4','--precision','bf16','--data-seed','23','--reference-threads','8',
            '--corpus','/work/natural-v2-1050m/source.json','--packed-corpus','/work/natural-v2-1050m/packed',
            '--dev-corpus','/work/natural-v2-1050m/dev.json','--warmup','4']
    lock=threading.Lock(); stop=threading.Event(); records=[]
    def persist():
        (args.output/'progress.json').write_text(json.dumps({'specification':specification,
            'records':records,'elapsed_seconds':time.time()-started},indent=2)+'\n')
    def run_phase(slot, name, arguments, steps, cohort, resume=None, decay=None, scientific=False):
        if stop.is_set() or time.time()>args.deadline_unix:
            raise TimeoutError('Campaign stopped/deadline reached')
        for file, value in specification['code_sha256'].items():
            if file_sha256(file)!=value: raise ValueError('Frozen scientific source changed: '+file)
        plan={'qualification_only':not scientific,'world':32,'visible_cores':list(range(slot*32,slot*32+32)),
              'training_memory_gib':288,'evaluation_memory_gib':288,'timeout_seconds':3600,
              'evaluation_cohort':cohort,'training_arguments':arguments,
              'phases':[{'run':name,'steps':steps,'pauses':0}]}
        if scientific: plan['scientific_manifest']=str(args.output/'launch-gates.json')
        if resume: plan['resume']=resume
        if decay:
            plan['phases'][0].update(decay_start=targets(decay[0]),decay_end=targets(decay[1]),branch=True)
        path=args.output/(name+'-plan.json');path.write_text(json.dumps(plan,indent=2)+'\n')
        subprocess.run(['python3','-m','scripts.run_phases','--plan',str(path),
                        '--output',str(args.output/(name+'-controller'))],check=True)
        report=json.loads((root/'runs'/name/'report.json').read_text())
        dev=json.loads((root/'runs'/(name+'-dev')/'report.json').read_text())
        return {'run':name,'checkpoint':f'{container_root}/runs/{name}/checkpoint.pt',
                'update':sum(r['updates'] for r in [report])+ (0 if not resume else 0),
                'real_tokens':report['total_real_targets'],'dev':dev['metrics'],
                'completed_unix':time.time(),'slot':slot,'state_digest':report['state_digest']}
    # Every worker owns a disjoint core group. A queue keeps both occupied.
    def execute_jobs(jobs, worker):
        queue=list(jobs);qlock=threading.Lock()
        def loop(slot):
            while not stop.is_set():
                with qlock:
                    if not queue:return
                    item=queue.pop(0)
                try: worker(slot,item)
                except BaseException:
                    stop.set();raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(loop,slot) for slot in [0,1]]
            for future in futures: future.result()
    trials=[]
    jobs=[(arm,index,lr) for index,lr in enumerate(specification['lrs']) for arm in specification['arms']]
    def tune(slot,item):
        arm,index,lr=item;name=f'{args.identity}-lr-{arm}-{index}'
        arguments=[*common,'--purpose','tuning','--seed','11','--repetitions',str(specification['arms'][arm]),
                   '--lr',str(lr),'--lr-warmup-tokens',str(targets(16)),
                   '--decay-start',str(targets(346)),'--decay-end',str(targets(384))]
        result=run_phase(slot,name,arguments,384,'tune')
        with lock:
            trials.append({'arm':arm,'lr':lr,'run':name,'status':'PASS',
                           'tune_weighted_nll':result['dev']['weighted_nll']})
            (args.output/'tuning.json').write_text(json.dumps(trials,indent=2)+'\n')
    try:
        execute_jobs(jobs,tune)
        selected={arm:min([t for t in trials if t['arm']==arm],key=lambda t:t['tune_weighted_nll'])['lr'] for arm in specification['arms']}
        (args.output/'selected-lrs.json').write_text(json.dumps({'selected_lrs':selected,'frozen_unix':time.time(),
            'criterion':'Minimum fixed 60/20/20 tuning DEV NLL; all candidates retained'},indent=2)+'\n')
        gates={k:True for k in ['immutable_sources','near_duplicate_audit','frozen_order_sha256',
             'lr_selected_at_this_batch','measured_budget','trainer_qualified']}
        gates.update(specification_sha256=file_sha256(args.output/'specification.json'),
                     selection_sha256=file_sha256(args.output/'selected-lrs.json'))
        (args.output/'launch-gates.json').write_text(json.dumps(gates,indent=2)+'\n')
        persist()
        def main_fit(slot,item):
            arm,seed=item;r=specification['arms'][arm]
            arguments=[*common,'--purpose','scientific','--seed',str(seed),'--repetitions',str(r),
                       '--lr',str(selected[arm]),'--lr-warmup-tokens',str(targets(40)),
                       '--checkpoint-every','384']
            previous=None;old_step=0
            phases=[(384,'trunk'),(1152,'trunk')]
            if r==2:phases += [(fork,'compute_trunk'),(q,'compute_endpoint')]
            phases += [(3456,'token_trunk'),(3840,'token_endpoint')]
            for step,role in phases:
                name=f'{args.identity}-{arm}-s{seed}-u{step}'
                decay=(fork,q) if role=='compute_endpoint' else ((3456,3840) if role=='token_endpoint' else None)
                result=run_phase(slot,name,arguments,step-old_step,'curve',resume=previous,decay=decay,scientific=True)
                result.update(arm=arm,seed=seed,update=step,role=role)
                if role!='compute_endpoint':previous=result['checkpoint'];old_step=step
                with lock:
                    records.append(result);persist()
                    print(json.dumps({'event':'curve_record','arm':arm,'seed':seed,'update':step,
                                      'role':role,'weighted_nll':result['dev']['weighted_nll']}),flush=True)
        jobs=[(arm,seed) for seed in specification['seeds'] for arm in specification['arms']]
        execute_jobs(jobs,main_fit)
        (args.output/'complete.json').write_text(json.dumps({'status':'COMPLETE_FIXED_RECURRENCE_CURVES',
             'elapsed_seconds':time.time()-started,'records':len(records),'ended_unix':time.time()},indent=2)+'\n')
    except BaseException as error:
        (args.output/'failure.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error),
            'elapsed_seconds':time.time()-started,'records':len(records)},indent=2)+'\n')
        raise


if __name__=='__main__':main()
