"""Two fixed untied replications and one annealed ordinary compute control."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
from compute_trainium.confirmation import matched_endpoint
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args()
    root=Path(os.environ['RESEARCH_ROOT']);croot=os.environ['RESEARCH_CONTAINER_ROOT']
    a.output.mkdir(exist_ok=False);started=time.time()
    load=lambda path:json.loads(path.read_text())
    if load(root/'mechanism-campaign-v2/queue/report.json')['status']!='COMPLETE_QUEUE':
        raise ValueError('Original campaign must finish before follow-up')
    if load(root/'mechanism-endpoint-cpu-v1/report.json')['status']!='PASS_THREE_NEW_ENDPOINT_CPU_AUDITS':
        raise ValueError('Independent original endpoint audits must pass')
    for arm in ['ordinary','shared','untied']:
        calibration=load(root/f'runs/confirmation-v3-calibration-{arm}/report.json')
        if (calibration['status']!='PASS_SHARDED_TRAINED_PATH_READOUT' or
                abs(calibration['scalar_calibration_delta_nats'])>.0001):
            raise ValueError('Sharded DEV template calibration must pass')
    selected=load(root/'untied-depth-v1/selected-lr.json')
    if selected['lr']!=.0003:raise ValueError('Frozen selected LR differs')
    proofs={s:load(root/f'runs/untied-initial-s{s}-cpu-001/report.json') for s in [47,71]}
    if any(v['status']!='PASS_FULL_SIZE_UNTIED_INITIALIZATION' or v['seed']!=s for s,v in proofs.items()):
        raise ValueError('Paired initial-function proofs missing')
    untied=load(root/'runs/untied-depth-v1-s23-u3456/report.json')
    ordinary=load(root/'runs/stable-cont-v2-ordinary-s23-u7680/report.json')
    flops=matched_endpoint(3840,untied['flops_per_global_step']['training_matmul_flops'],ordinary['flops_per_global_step']['training_matmul_flops'])
    old=root.parent/'natural-v2-1050m';extended=root/'natural-append-3250m'
    prefix_old=np.load(old/'packed/prefix_counts.npy',mmap_mode='r')
    prefix_new=np.load(extended/'packed/prefix_counts.npy',mmap_mode='r')
    count=lambda u,extension=False:int((prefix_new if extension else prefix_old)[u*256].sum())
    endpoint=flops['updates'];stable=flops['stable_update']
    # Measured rates include accumulation, clipping and periodic checkpointing.
    training_seconds=(2*count(3840)/untied['warm_global_real_targets_per_second']+
        (count(endpoint,True)-count(3456,True))/ordinary['warm_global_real_targets_per_second'])
    budget={'observed_untied_real_targets_per_second':untied['warm_global_real_targets_per_second'],
        'observed_ordinary_real_targets_per_second':ordinary['warm_global_real_targets_per_second'],
        'training_seconds':training_seconds,'training_margin_factor':1.3,
        'six_fresh_training_and_dev_overheads_seconds':6*230,
        'protected_test_seconds':1800,'protected_reporting_seconds':600}
    required=training_seconds*1.3+6*230+1800+600
    budget['required_seconds']=required;budget['available_seconds']=a.deadline_unix-time.time()
    write_state(a.output/'measured-budget.json',budget)
    if time.time()+required>a.deadline_unix:raise TimeoutError('Complete follow-up plus confirmation budget does not fit')
    files=['scripts/train_distributed.py','compute_trainium/model.py','compute_trainium/titan.py',
        'compute_trainium/recurrence.py','compute_trainium/preln_recurrence.py','compute_trainium/continuation.py',
        'compute_trainium/training.py','compute_trainium/data.py','scripts/run_confirmation_followups.py','scripts/evaluate_trained_path.py',
        'docs/CONFIRMATION_FOLLOWUP_20261023.md']
    spec={'identity':'confirmation-followups-v3','frozen_unix':time.time(),'deadline_unix':a.deadline_unix,
        'untied_seeds':[47,71],'untied_selected_lr':.0003,'data_order_seed':23,
        'flop_match':flops,'ordinary_targets':count(endpoint,True),'ordinary_stable_targets':count(stable,True),
        'untied_targets':count(3840),'untied_initial_proofs':proofs,'budget':budget,
        'source_sha256':{f:file_sha256(f) for f in files},
        'original_manifest_sha256':file_sha256(old/'manifest.json'),
        'extension_manifest_sha256':file_sha256(extended/'manifest.json'),
        'scope':'Two fresh untied seeds; discovery seed23 reused. One annealed estimated-matmul matched ordinary endpoint. No new mechanism, LR search or TEST access.'}
    write_state(a.output/'specification.json',spec)
    gates={k:True for k in ['immutable_sources','near_duplicate_audit','frozen_order_sha256','lr_selected_at_this_batch','measured_budget','trainer_qualified']}
    write_state(a.output/'launch-gates.json',gates);records=[]
    # Resource control first, then independent seeds in their declared order.
    for arm,seed,updates,parent,extension in [
        ('ordinary',23,[stable,endpoint],'/work/runs/natural-curve-v1-ordinary-s23-u3840/stable-u3456.pt',True),
        ('untied',47,[3456,3840],None,False),('untied',71,[3456,3840],None,False)]:
        previous=parent;prior=3456 if extension else 0
        corpus=f'{croot}/natural-append-3250m' if extension else '/work/natural-v2-1050m'
        end=endpoint if extension else 3840;start=stable if extension else 3456
        common=['--size','small','--architecture','ordinary' if extension else 'preln','--repetitions','2',
            '--context','1024','--batch','2','--accumulation','2','--precision','bf16','--optimizer','plain',
            '--seed',str(seed),'--data-seed','23','--reference-threads','8','--corpus',corpus+'/source.json',
            '--packed-corpus',corpus+'/packed','--dev-corpus','/work/natural-v2-1050m/dev.json',
            '--lr','.001' if extension else '.0003','--lr-warmup-tokens',str(count(40,extension)),
            '--decay-start',str(count(start,extension)),'--decay-end',str(count(end,extension)),
            '--checkpoint-every','384','--warmup','4','--purpose','scientific']
        common+=['--allow-corpus-extension','--extend-stable-schedule'] if extension else ['--untied-depth']
        for update in updates:
            if time.time()+2400>a.deadline_unix:raise TimeoutError('Protected final confirmation reserve')
            if (root/'hold-scientific.json').exists():raise RuntimeError('Explicit scientific launch hold')
            for f,sha in spec['source_sha256'].items():
                if file_sha256(f)!=sha:raise ValueError('Frozen follow-up code or protocol changed')
            label='ordinary-flop' if extension else 'untied'
            name=f'confirmation-v3-{label}-s{seed}-u{update}'
            plan={'qualification_only':False,'scientific_manifest':str(a.output/'launch-gates.json'),
                'world':64,'evaluation_world':64,'training_memory_gib':288 if extension else 320,'evaluation_memory_gib':224,
                'timeout_seconds':5400,'evaluation_cohort':'curve','resume':previous,'training_arguments':common,
                'phases':[{'run':name,'steps':update-prior,'pauses':0}]}
            path=a.output/(name+'-plan.json');write_state(path,plan)
            # This evaluator must first reproduce each scalar architecture
            # template on DEV. Parallelize readout without changing its path,
            # token contexts, targets, weights or numerical objective.
            env=dict(os.environ,RESEARCH_RUN=name,RESEARCH_ENTRY_MODULE='scripts.train_distributed',
                RESEARCH_DDP_RANKS='64',NEURON_RT_VISIBLE_CORES=','.join(map(str,range(64))),RESEARCH_TIMEOUT_SECONDS='5400')
            subprocess.run(['docker','update','--memory',str(plan['training_memory_gib']*1024**3),'autonomous-compute-neuron233'],check=True,capture_output=True)
            command=[os.sys.executable,'-m','scripts.supervise',*common,'--steps',str(update-prior),'--pauses','0','--save']
            if previous:command+=['--resume',previous]
            subprocess.run(command,env=env,check=True)
            subprocess.run(['docker','update','--memory',str(224*1024**3),'autonomous-compute-neuron233'],check=True,capture_output=True)
            env.update(RESEARCH_RUN=name+'-dev',RESEARCH_ENTRY_MODULE='scripts.evaluate_trained_path',RESEARCH_TIMEOUT_SECONDS='900')
            subprocess.run([os.sys.executable,'-m','scripts.supervise','--checkpoint',f'{croot}/runs/{name}/checkpoint.pt',
                '--corpus','/work/natural-v2-1050m/dev.json','--repetitions','1' if extension else '2','--split','dev'],env=env,check=True)
            train=load(root/'runs'/name/'report.json');dev=load(root/'runs'/(name+'-dev')/'report.json')
            if train['status']!='PASS_PACKED_TRAINING_PHASE' or dev['status']!='PASS_SHARDED_TRAINED_PATH_READOUT':
                raise ValueError('Follow-up train or calibrated readout failed')
            if (train['total_real_targets']!=count(update,extension) or not train['all_replica_model_optimizer_states_equal']):
                raise AssertionError('Follow-up exposure or replica state differs')
            if arm=='untied' and train['initial_model_digest']!=proofs[seed]['full_untied_initial_digest']:
                raise AssertionError('Paired untied initialization differs')
            if previous is not None and not train['recovered_model_optimizer_digest']:
                raise AssertionError('Full optimizer recovery missing')
            previous=f'{croot}/runs/{name}/checkpoint.pt';prior=update
            if extension and update==endpoint and train['total_training_matmul_flops']!=flops['candidate_flops']:
                raise AssertionError('Compute-matched total work differs')
            records.append({'arm':arm,'seed':seed,'update':update,'real_tokens':train['total_real_targets'],
                'metrics':dev['metrics'],'checkpoint':previous,'run':name,'state_digest':train['state_digest'],
                'training_matmul_flops':train['total_training_matmul_flops'],
                'warm_real_targets_per_second':train['warm_global_real_targets_per_second']})
            write_state(a.output/'progress.json',records)
            print(json.dumps({'event':'confirmation_followup','arm':arm,'seed':seed,'update':update,'nll':dev['metrics']['weighted_nll']}),flush=True)
    write_state(a.output/'report.json',{'status':'COMPLETE_CONFIRMATION_FOLLOWUPS','specification':spec,'records':records,'elapsed_seconds':time.time()-started})


if __name__=='__main__':main()
