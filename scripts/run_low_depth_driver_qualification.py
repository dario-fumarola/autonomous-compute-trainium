"""Exercise finite scientific-driver segments and an independent cooldown branch."""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from compute_trainium.execution_isolation import managed_running,metadata,cancel_worker_container
from compute_trainium.owned_native_compact import create_compact_worker,reclaim_host
from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig
from compute_trainium.parcae_scientific_schedule import update_work,source_endpoint
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,file_sha256


def released_half(records,arm):
    return any(r['arm']==arm and r['precision']=='fp16' and r['accumulation']==16
               and r['passes_recovery_gate'] for r in records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    out=a.root/'low-depth-driver-qualification-v1';out.mkdir(exist_ok=True)
    public=Path('results/low_depth_driver_qualification_20261024');public.mkdir(parents=True,exist_ok=True)
    profile=Path('results/parcae_low_depth_qualification_20261024/attempt4')
    plan_file=out/'plan.json'
    if not plan_file.exists():
        reference=a.root/'parcae-low-depth-qualification-v1'
        q=json.loads((reference/'plan.json').read_text());config=ParcaeCandidateConfig(**q['model'])
        native=json.loads((profile/'native-progress.json').read_text())['records']
        audits=json.loads((profile/'trained-audit-progress.json').read_text())['records']
        basis=[]
        for arm in ['ordinary','poisson2']:
            n=next(r for r in native if r['arm']==arm and r['precision']=='fp16' and r['accumulation']==16)
            c=next(r for r in audits if r['arm']==arm and r['precision']=='fp16' and r['accumulation']==16)
            if not n['passes_recovery_gate'] or c['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT':
                raise AssertionError('Complete per-arm operator/training qualification required')
            basis.append({'arm':arm,'native':n,'independent_cpu':c})
        write_state(out/'qualification-basis.json',{'records':basis});write_state(public/'qualification-basis.json',{'records':basis})
        sources=dict(json.loads((a.root/'parcae-low-depth-qualification-v4/native-plan.json').read_text())['source_sha256'])
        for file in ['scripts/run_low_depth_driver_qualification.py','scripts/train_low_depth_scientific.py',
                     'compute_trainium/parcae_scientific_state.py','compute_trainium/parcae_scientific_schedule.py',
                     'tests/test_parcae_scientific_state.py','tests/test_parcae_scientific_schedule.py',
                     'docs/LOW_DEPTH_DRIVER_QUALIFICATION_20261024.md']:
            sources[file]=file_sha256(file)
        packed=PackedCorpus.from_mapped(a.root/'natural-append-6500m-v2/packed',1024,23,q['source_corpus_sha256'])
        initials={};cases={}
        for arm in ['ordinary','poisson2']:
            kind='ordinary' if arm=='ordinary' else 'family'
            initials[kind]={**q['initial_states'][kind],'file':'../parcae-low-depth-qualification-v1/'+q['initial_states'][kind]['file']}
            table='../parcae-low-depth-qualification-v1/'+arm+'-a16-depths.npy'
            depths=np.load(out/table,mmap_mode='r')
            with torch.device('meta'):
                model=ParcaeCandidate(config,recurrent=arm!='ordinary',initialize=False)
            optimizer=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')
            work=update_work(config,depths,recurrent=arm!='ordinary',context=1024,world=32,optimizer_per_replica=optimizer)
            np.save(out/(arm+'-work.npy'),work)
            for suffix,end in [('main',40),('short',30)]:
                case={'arm':arm,'seed':1009,'state_seed':2009,'initial_state':kind,
                    'parameters':q['initial_states'][kind]['parameters'],'peak_lr':6e-4,
                    'warmup_targets':source_endpoint(packed.prefix_counts,4,512)['real_targets'],
                    'cooldown_start_update':20,'cooldown_start_targets':source_endpoint(packed.prefix_counts,20,512)['real_targets'],
                    'total_updates':end,'end_targets':source_endpoint(packed.prefix_counts,end,512)['real_targets'],
                    'depth_table':table,'depth_table_sha256':file_sha256(out/table),
                    'work_table':arm+'-work.npy','work_table_sha256':file_sha256(out/(arm+'-work.npy')),
                    'checkpoint_updates':[20,40] if suffix=='main' else [30],
                    'segment_ends':[20,40] if suffix=='main' else [30]}
                if suffix=='short':case['parent']={'case_id':arm+'-main','update':20}
                cases[arm+'-'+suffix]=case
        plan={'identity':'low-depth-driver-qualification-v1','registered_unix':time.time(),
            'purpose':'low_depth_driver_integration_qualification','launch_gates_complete':True,
            'source_sha256':sources,'worker_source_sha256':{k:v for k,v in sources.items() if k.startswith(('compute_trainium/','scripts/','configs/'))},
            'basis_sha256':file_sha256(out/'qualification-basis.json'),'model':q['model'],'precision':'fp16',
            'accumulation':16,'routing_seed':271828,'initial_states':initials,'cases':cases,
            'data':{'corpus':'natural-append-6500m-v2/source.json','packed':'natural-append-6500m-v2/packed',
                'corpus_sha256':q['source_corpus_sha256'],'manifest_sha256':q['packed_manifest_sha256'],
                'packing_fingerprint':q['packing_fingerprint']},
            'seconds_limit':1200,'phase_seconds_limit':600,'research_optimizer_updates':0,'test_accessed':False,
            'scope':'Qualification-only new driver, two model arms,20+20+10 actual updates each. No scientific initialization/fitting or held-out scoring; exact registered schedule/branch/full-state/source/work checks.'}
        write_state(plan_file,plan);write_state(public/'plan.json',plan)
    plan=json.loads(plan_file.read_text());plan_sha=file_sha256(plan_file)
    def frozen():
        if file_sha256(plan_file)!=plan_sha or any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):
            raise AssertionError('Driver qualification source/plan changed')
    frozen()
    if a.prepare_only:return
    if (out/'report.json').exists():raise FileExistsError('No silent driver qualification rerun')
    started=time.time();deadline=started+plan['seconds_limit'];active={};records=[];costs=[]
    lock=threading.Lock();failed=threading.Event()
    def phase(arm,half,label,case,end,parent=None):
        frozen();reclaim_host()
        identity=f'low-depth-driver-v1-{arm}-{label}'
        container=create_compact_worker(a.root,identity,half);process=None
        with lock:active[identity]=container
        try:
            limit=min(deadline,time.time()+600)
            env=dict(os.environ,RESEARCH_ROOT=str(a.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                RESEARCH_RUN=identity,RESEARCH_OWNER=identity,RESEARCH_CONTAINER=container,
                RESEARCH_ENTRY_MODULE='scripts.train_low_depth_scientific',RESEARCH_DDP_RANKS='32',
                NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,half*32+32))),
                RESEARCH_TIMEOUT_SECONDS=str(max(1,int(limit-time.time()))))
            command=[os.sys.executable,'-u','-m','scripts.supervise_isolated','--root','/work/instance-nvme',
                '--plan','/work/instance-nvme/low-depth-driver-qualification-v1/plan.json','--plan-sha',plan_sha,
                '--case',case,'--segment-end',str(end)]
            if parent:
                command+=['--resume','/work/instance-nvme/runs/'+parent['identity']+'/checkpoint-u20.pt','--resume-sha',parent['sha']]
            with (out/(identity+'.log')).open('x') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if failed.is_set() or time.time()>limit:raise RuntimeError('Driver qualification paired stop/deadline')
                time.sleep(5)
            if process.returncode:raise RuntimeError('Driver native phase failed: '+identity)
            report=json.loads((a.root/'runs'/identity/'report.json').read_text())
            expected=20 if label!='short' else 10
            if (report['status']!='COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT' or not report['qualification_only']
                or report['research_optimizer_updates']!=0 or report['actual_optimizer_calls_per_replica']!=expected
                or any(not r['all_complete_replica_states_equal'] for r in report['checkpoint_records'])):
                raise AssertionError('Driver qualification count/purpose/replica gate failed')
            if parent and (not report['complete_state_on_load_exact'] or report['recovered_state_digest']!=parent['digest']):
                raise AssertionError('Driver complete state/registered branch recovery differs')
            with lock:records.append({'identity':identity,**report});write_state(public/'progress.json',{'records':records})
            if label=='first':
                checkpoint=next(r for r in report['checkpoint_records'] if r['update']==20)
                return {'identity':identity,'sha':checkpoint['checkpoint_sha256'],'digest':checkpoint['full_state_digest']}
        finally:
            cancel_worker_container(container,identity,'driver_qualification_phase_complete')
            if process and process.poll() is None:
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.terminate()
            closure=a.root/'snapshots'/identity/'closure.json'
            with lock:
                active.pop(identity,None)
                costs.append({'identity':identity,**json.loads(closure.read_text())} if closure.exists() else {'identity':identity,'closure_missing':True})
    def role(arm,half):
        try:
            final='fixed1' if half==0 else 'poisson2'
            while True:
                if failed.is_set() or time.time()>deadline:raise RuntimeError('Driver admission paired stop/deadline')
                current=json.loads((profile/'native-progress.json').read_text())['records']
                occupied=any((metadata(c)['Config'].get('Labels') or {}).get('ac-trainium-half') in [None,str(half)] for c in managed_running())
                if released_half(current,final) and not occupied:break
                time.sleep(5)
            first=phase(arm,half,'first',arm+'-main',20)
            phase(arm,half,'continued',arm+'-main',40,first)
            phase(arm,half,'short',arm+'-short',30,first)
        except BaseException:failed.set();raise
    try:
        write_state(public/'status.json',{'status':'RUNNING_OR_WAITING_RELEASED_DRIVER_QUALIFICATION','maximum_chips':16,'started_unix':started})
        with cf.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(role,'ordinary',1),pool.submit(role,'poisson2',0)]
            for future in cf.as_completed(futures):future.result()
        frozen();result={'status':'PASS_FINITE_LOW_DEPTH_DRIVER_AND_BRANCH_RECOVERY','records':records}
    except BaseException as e:
        failed.set();result={'status':'STOPPED_LOW_DEPTH_DRIVER_QUALIFICATION','records':records,'error':str(e)};raise
    finally:
        for identity,container in list(active.items()):cancel_worker_container(container,identity,'driver_qualification_stop')
        result.update(wall_seconds=time.time()-started,research_optimizer_updates=0,test_accessed=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0,'observed_unix':time.time()})
        write_state(public/'execution-costs.json',{'native_attempts':costs,'wall_seconds':time.time()-started,
            'scope':'All qualification-only driver updates, cold launches, checkpoints/resumes and failures. Count overlapping intervals once for instance opportunity cost.'})


if __name__=='__main__':main()
