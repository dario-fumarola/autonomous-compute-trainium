"""Finite three-seed3B fixed-recurrence development comparison; never reads TEST."""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from compute_trainium.depth_analysis import family_seed_interval
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    a.output.mkdir(exist_ok=True);root=a.root;public=Path('results/mature_fixed_20261024');public.mkdir(exist_ok=True)
    started=time.time();lock=threading.Lock();stop=threading.Event();active=[];audit=None
    state={'status':'GATING','stages':{},'controller_pid':os.getpid()}
    cutoff=dt.datetime(2026,10,24,10,15,tzinfo=dt.timezone.utc).timestamp()
    deadline=dt.datetime(2026,10,24,11,tzinfo=dt.timezone.utc).timestamp()
    def update(name,row):
        with lock:
            state['stages'][name]=row;state['heartbeat_unix']=time.time();write_state(a.output/'state.json',state)
    def heartbeat():
        while not (a.output/'report.json').exists():update('liveness',{'status':'ALIVE','unix':time.time()});time.sleep(20)
    def frozen():
        if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Saved scientific source changed')
    def summary_unlocked():
        rows=[]
        for model in inventory:
            depth=1 if model['arm']=='ordinary' else 2
            path=root/'runs'/f"{model['identity']}-dev-r{depth}"/'report.json'
            closure=root/'snapshots'/f"{model['identity']}-dev-r{depth}"/'closure.json'
            if not path.exists() or not closure.exists() or json.loads(closure.read_text())['exit_code']:continue
            r=json.loads(path.read_text());rows.append({**model,'metrics':r['metrics'],
                'training_real_targets':r['training_real_tokens']})
        curves=[]
        for seed in [103,211,307]:
            for arm in ['ordinary','fixed']:
                depth=1 if arm=='ordinary' else 2
                for target in [7680,10368,11520]:
                    name=f'mature-fixed-v1-{arm}-s{seed}-u{target}-dev-r{depth}'
                    path=root/'runs'/name/'report.json';closure=root/'snapshots'/name/'closure.json'
                    if path.exists() and closure.exists() and json.loads(closure.read_text())['exit_code']==0:
                        r=json.loads(path.read_text());curves.append({'arm':arm,'seed':seed,'update':target,
                            'schedule':'annealed' if target==11520 else 'stable','real_targets':r['training_real_tokens'],'metrics':r['metrics']})
        result={'learning_curve':curves,'status':'COMPLETE_DEVELOPMENT_READOUT' if len(rows)==9 else 'PARTIAL_DEVELOPMENT_READOUT',
                'rows':rows,'scope':'Registered3B development comparison. Same original seeds/order and consumed DEV; no new TEST or terminal significance claim.'}
        if len(rows)==9:
            lookup={(r['arm'],r['seed'],r['endpoint']):r['metrics']['weighted_nll'] for r in rows}
            result['contrasts']={key:family_seed_interval({str(s):lookup['ordinary',s,end]-lookup['fixed',s,'3B'] for s in [103,211,307]},comparisons=2)
                for key,end in [('compute_matched','matched'),('source_matched','3B')]}
        write_state(public/'progress.json',result)
        if len(rows)==9:write_state(public/'summary.json',result)
    def summary():
        with lock:summary_unlocked()
    def native(name,module,args,half,evaluation=False):
        if audit and audit.poll() is not None and audit.returncode:raise RuntimeError('Independent audit failed')
        frozen();existing=root/'runs'/name/'report.json';closure=root/'snapshots'/name/'closure.json'
        if existing.exists():
            if not a.resume:raise FileExistsError('Completed stage requires explicit --resume')
            launch=json.loads((closure.parent/'launch.json').read_text())
            if not closure.exists() or json.loads(closure.read_text())['exit_code'] or launch['arguments']!=args:
                raise ValueError('Existing stage differs from registered successful launch')
            for f,sha in plan['source_sha256'].items():
                if f.endswith('.py') and f in launch['source_sha256'] and launch['source_sha256'][f]!=sha:
                    raise AssertionError('Existing stage used different dependencies')
            return json.loads(existing.read_text())
        if (root/'runs'/name).exists():raise RuntimeError('Incomplete stage needs explicit preserved-checkpoint recovery: '+name)
        limit=deadline if evaluation else cutoff
        if stop.is_set() or time.time()+120>limit:raise TimeoutError('No registered phase time remains')
        cores=list(range(half*32,(half+1)*32));cores=cores[::4] if evaluation else cores
        env=dict(os.environ,RESEARCH_ROOT=str(root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',RESEARCH_RUN=name,
                 RESEARCH_ENTRY_MODULE=module,RESEARCH_DDP_RANKS=str(len(cores)),NEURON_RT_VISIBLE_CORES=','.join(map(str,cores)),
                 RESEARCH_TIMEOUT_SECONDS=str(min(5400,int(limit-time.time()))))
        with (a.output/(name+'.log')).open('w') as log:
            proc=subprocess.Popen([os.sys.executable,'-u','-m','scripts.supervise',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
        with lock:active.append(proc)
        update(name,{'status':'RUNNING','pid':proc.pid,'half':half,'started_unix':time.time()})
        try:code=proc.wait(timeout=max(1,limit-time.time()+30))
        finally:
            if proc.poll() is not None:
                with lock:active.remove(proc)
        if code:raise RuntimeError('Native stage failed: '+name)
        r=json.loads(existing.read_text());update(name,{'status':'COMPLETE','ended_unix':time.time(),'result':r['status']})
        return r
    def evaluate(identity,arm,half):
        depth=1 if arm=='ordinary' else 2
        args=['--checkpoint',f'/work/instance-nvme/runs/{identity}/checkpoint.pt',
              '--corpus','/work/instance-nvme/variable-depth-fresh-v1/dev.json','--split','dev','--cohort','curve',
              '--repetitions',str(depth)]
        native(identity+f'-dev-r{depth}','scripts.evaluate_depth_curve',args,half,True);summary()
    def train(arm,seed,parent,prior,target,decay,end,half):
        name=f'mature-fixed-v1-{arm}-s{seed}-u{target}'
        args=base+['--purpose','scientific','--seed',str(seed),'--depth-policy',arm,
            '--lr',str(.0006 if arm=='ordinary' else .001),'--lr-warmup-tokens',str(count(96)),
            '--decay-start',str(count(decay)),'--decay-end',str(count(end)),
            '--steps',str(target-prior),'--resume',parent,'--extend-stable-schedule']
        r=native(name,'scripts.train_variable_depth',args,half)
        if r['status']!='PASS_PACKED_TRAINING_PHASE' or not r['all_replica_model_optimizer_states_equal']:
            raise AssertionError('Full distributed phase failed qualification')
        if r['total_real_targets']!=count(target) or r['total_training_matmul_flops']!=target*plan['step_flops'][arm]:
            raise AssertionError('Training counters differ from the frozen source/work plan')
        return name,f'/work/instance-nvme/runs/{name}/checkpoint.pt'
    def fit(item,half):
        arm,seed=item;parent=parents[arm,str(seed)];prior=3456
        decay,end=(17541,19490) if arm=='ordinary' else (10368,11520)
        for target in [7680,10368]:
            name,parent=train(arm,seed,parent,prior,target,decay,end,half);prior=target;evaluate(name,arm,half)
        stable=parent
        name,_=train(arm,seed,stable,10368,11520,10368,11520,half);evaluate(name,arm,half)
        if arm=='ordinary':
            name,parent=train(arm,seed,stable,10368,17541,17541,19490,half)
            name,_=train(arm,seed,parent,17541,19490,17541,19490,half);evaluate(name,arm,half)
    try:
        q=json.loads((root/'source-native-equivalence-v2/report.json').read_text())
        if q['status']!='PASS_NATIVE_APPEND_SOURCE_CONTINUATION':raise ValueError('Native source migration not qualified')
        copies=json.loads((a.output/'state-copy-qualification.json').read_text())
        if copies['status']!='PASS_ALL_SIX_CPU_SOURCE_STATE_COPIES':raise ValueError('Stable state copies incomplete')
        if json.loads((root/'natural-append-6500m-v2/independent-audit.json').read_text())['status']!='PASS_INDEPENDENT_SECOND_EXTENSION_AUDIT':
            raise ValueError('Source corpus audit failed')
        prefix=np.load(root/'natural-append-6500m-v2/packed/prefix_counts.npy',mmap_mode='r')
        def count(update):return int(prefix[update*256].sum())
        parents={(r['arm'],str(r['seed'])):r['container_checkpoint'] for r in copies['records']}
        base=['--corpus','/work/instance-nvme/natural-append-6500m-v2/source.json',
              '--packed-corpus','/work/instance-nvme/natural-append-6500m-v2/packed',
              '--dev-corpus','/work/instance-nvme/variable-depth-fresh-v1/dev.json','--context','1024',
              '--batch','4','--accumulation','2','--architecture','preln','--precision','bf16','--optimizer','fused',
              '--recipe','pretraining','--data-seed','23','--reference-threads','4','--warmup','4','--save','--checkpoint-every','1024']
        inventory=[{'identity':f'mature-fixed-v1-{arm}-s{seed}-u{target}','arm':arm,'seed':seed,'endpoint':endpoint,
                    'parameters':150436608,'allowed_depths':[1 if arm=='ordinary' else 2]}
                   for seed in [103,211,307] for arm,endpoint,target in [('ordinary','3B',11520),('fixed','3B',11520),('ordinary','matched',19490)]]
        if (a.output/'plan.json').exists():
            if not a.resume:raise FileExistsError('Existing campaign requires --resume')
            plan=json.loads((a.output/'plan.json').read_text());frozen()
        else:
            old=json.loads((root/'variable-depth-campaign-v2/plan.json').read_text());sources=old['source_sha256'].copy()
            for f in ['docs/MATURE_FIXED_PROTOCOL_20261024.md','scripts/run_mature_fixed_campaign.py',
                      'scripts/monitor_mature_fixed.py','scripts/migrate_checkpoint.py','scripts/supervise.py','scripts/run_native.sh']:
                sources[f]=file_sha256(f)
            plan={'identity':'mature-fixed-v1','registered_unix':time.time(),'training_cutoff_unix':cutoff,'deadline_unix':deadline,
                  'seeds':[103,211,307],'source_sha256':sources,'step_flops':{'ordinary':286044821913600,'fixed':483956914913280},
                  'source_manifest_sha256':file_sha256(root/'natural-append-6500m-v2/manifest.json'),
                  'dev_sha256':file_sha256(root/'variable-depth-fresh-v1/dev.json'),
                  'parent_copy_receipt_sha256':file_sha256(a.output/'state-copy-qualification.json'),
                  'native_source_qualification_sha256':file_sha256(root/'source-native-equivalence-v2/report.json'),
                  'endpoints':{'data_updates':11520,'data_targets':count(11520),'matched_updates':19490,'matched_targets':count(19490)},
                  'scope':'Three-seed bounded mature-exposure DEV screen. No TEST. Prior seeds/recipe reused; fresh terminal confirmation remains required.',
                  'budget':{'qualified_100_warm_targets_per_second':{'ordinary':900268.0731684831,'fixed':577029.4499780975},
                            'predicted_training_node_seconds':12978.85206274654,'predicted_with_30_percent_and_90_minutes':22272.507681570503}}
            if time.time()+plan['budget']['predicted_with_30_percent_and_90_minutes']>deadline:
                raise TimeoutError('Measured entire campaign cannot fit the registered hard budget')
            write_state(a.output/'plan.json',plan);write_state(public/'plan.json',plan)
        for row in copies['records']:
            if file_sha256(Path(row['checkpoint']))!=row['checkpoint_sha256']:raise AssertionError('Derived stable source copy changed')
        if file_sha256(root/'natural-append-6500m-v2/manifest.json')!=plan['source_manifest_sha256'] or file_sha256(root/'variable-depth-fresh-v1/dev.json')!=plan['dev_sha256']:
            raise AssertionError('Registered corpus/panel changed')
        write_state(a.output/'models.json',inventory);write_state(public/'models.json',inventory)
        if a.validate_only:
            write_state(a.output/'validation.json',{'status':'PASS_REGISTERED_PLAN_GATES_NO_NATIVE_LAUNCH','unix':time.time(),
                'source_fingerprints_and_six_parent_copy_hashes_checked':True,'plan_sha256':file_sha256(a.output/'plan.json')})
            return
        audit_args=[os.sys.executable,'-u','-m','scripts.audit_variable_depth','--root',str(root),'--models',str(a.output/'models.json'),
                    '--corpus',str(root/'variable-depth-fresh-v1/dev.json'),'--output',str(a.output/'independent-cpu'),'--deadline-unix',str(deadline-60)]
        with (a.output/'independent-cpu.log').open('a') as log:
            audit=subprocess.Popen(audit_args,env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_state(a.output/'independent-cpu-process.json',{'pid':audit.pid,'started_unix':time.time()})
        state['status']='RUNNING';threading.Thread(target=heartbeat,daemon=True).start()
        pending=queue.Queue()
        for item in [('ordinary',103),('fixed',103),('ordinary',211),('ordinary',307),('fixed',211),('fixed',307)]:pending.put(item)
        def consume(half):
            while not stop.is_set():
                try:item=pending.get_nowait()
                except queue.Empty:return
                try:fit(item,half)
                except BaseException:stop.set();raise
        with cf.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(consume,h) for h in [0,1]]
            try:
                for future in cf.as_completed(futures):future.result()
            except BaseException:
                stop.set()
                if any(proc.poll() is None for proc in active):subprocess.run(['docker','stop','--time','5','autonomous-compute-neuron233'],capture_output=True,timeout=25)
                raise
        audit.wait(timeout=max(1,deadline-time.time()-30))
        if audit.returncode:raise RuntimeError('Independent endpoint audit failed')
        frozen();summary();write_state(public/'independent-cpu-verification.json',json.loads((a.output/'independent-cpu/report.json').read_text()))
        state['status']='COMPLETE_MATURE_FIXED_DEVELOPMENT_CAMPAIGN';update('closure',{'status':'COMPLETE'})
        report={'status':state['status'],'wall_seconds':time.time()-started,'native_endpoints':9,'test_accessed':False}
        write_state(a.output/'report.json',report);write_state(public/'report.json',report)
    except BaseException as error:
        stop.set()
        if any(proc.poll() is None for proc in active):subprocess.run(['docker','stop','--time','5','autonomous-compute-neuron233'],capture_output=True,timeout=25)
        if audit and audit.poll() is None:
            os.killpg(audit.pid,signal.SIGTERM)
            try:audit.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(audit.pid,signal.SIGKILL);audit.wait(timeout=10)
        report={'status':'STOPPED_MATURE_FIXED_DEVELOPMENT_CAMPAIGN','error':str(error),'wall_seconds':time.time()-started,'test_accessed':False}
        write_state(a.output/'report.json',report);write_state(public/'report.json',report);raise


if __name__=='__main__':main()
