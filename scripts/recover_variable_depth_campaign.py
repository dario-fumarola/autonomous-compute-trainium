"""Explicit recovery of completed stages, preserving the saved scientific freeze.

The original controller is left untouched. This operational amendment ignores
unrelated added files, budgets only pending work, adopts verified CPU audits
and preserves an existing TEST release. Occupied failed native identities stop
for explicit operator handling; they are never overwritten or blindly retried.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from compute_trainium.depth_analysis import SEEDS
from compute_trainium.preln_recurrence import PreNormRecurrenceConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.recovery_support import (ExistingAudit,archive_partial_audit,completed_stage,
    live_model_workers,preserve_test_release,process_matches,remaining_budget,verify_saved_sources)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--fresh-panel',type=Path,required=True)
    p.add_argument('--resume',action='store_true',default=True)
    p.add_argument('--inspect',action='store_true',help='Read-only source/remaining-work audit; starts nothing')
    a=p.parse_args();root=Path(os.environ['RESEARCH_ROOT']);croot=os.environ['RESEARCH_CONTAINER_ROOT']
    if not a.output.is_dir():raise FileNotFoundError('Recovery needs an existing campaign')
    saved_plan=json.loads((a.output/'plan.json').read_text())
    verified=verify_saved_sources(Path.cwd(),saved_plan)
    prefix_for_budget=np.load(root/'natural-append-3250m/packed/prefix_counts.npy',mmap_mode='r')
    estimate=remaining_budget(saved_plan,root,lambda update:int(prefix_for_budget[update*256].sum()))
    if a.inspect:
        print(json.dumps({'status':'READ_ONLY_RECOVERY_INSPECTION','sources':verified,'estimate':estimate},indent=2));return
    previous_state=json.loads((a.output/'state.json').read_text())
    old_pid=previous_state['controller_pid']
    if any(process_matches(old_pid,module,a.output) for module in ['scripts.run_variable_depth_campaign','scripts.recover_variable_depth_campaign']):
        raise RuntimeError('Original controller is still live; do not restart it')
    if live_model_workers():raise RuntimeError('Native handles remain live: observe those workers before recovery')
    occupied=[n for n in estimate['pending_stages'] if any(p.exists() for p in
              [root/'runs'/n,root/'snapshots'/n,root/'logs'/(n+'.log')])]
    if occupied:raise RuntimeError('Preserved occupied attempts require explicit operator recovery: '+repr(occupied))
    deadline=dt.datetime(2026,10,24,4,15,tzinfo=dt.timezone.utc).timestamp()
    training_cutoff=dt.datetime(2026,10,24,3,0,tzinfo=dt.timezone.utc).timestamp()
    if time.time()+estimate['remaining_wall_seconds_upper']>deadline or (
          estimate['pending_training_job_seconds'] and time.time()+estimate['remaining_training_wall_seconds_upper']>training_cutoff):
        raise TimeoutError('Only remaining work was budgeted, but it does not fit the original execution window')
    event=a.output/'recovery-events'/str(time.time_ns());event.mkdir(parents=True)
    write_state(event/'previous-state.json',previous_state)
    write_state(event/'amendment.json',{'purpose':'ORCHESTRATION_ONLY_RECOVERY','saved_plan_sha256':file_sha256(a.output/'plan.json'),
        'recovery_controller_sha256':file_sha256(__file__),'support_sha256':file_sha256('scripts/recovery_support.py'),
        'remaining_work_estimate':estimate,'scientific_sources':verified,
        'scope':'No model, source, LR, schedule, inference depth, outcome or statistical decision changes.'})
    old_report=a.output/'report.json'
    if old_report.exists():
        if json.loads(old_report.read_text())['status']=='COMPLETE_VARIABLE_DEPTH_CAMPAIGN':
            print('Campaign already complete; no restart');return
        old_report.rename(event/'previous-report.json')
    started=time.time();lock=threading.Lock();background_processes=[];state={'status':'WAITING_QUALIFICATION','stages':{},'controller_pid':os.getpid()}
    def state_update(key,value):
        with lock:
            state['stages'][key]=value;state['heartbeat_unix']=time.time();write_state(a.output/'state.json',state)
    def heartbeat():
        while not (a.output/'report.json').exists():
            state_update('controller_liveness',{'status':'ALIVE','unix':time.time()});time.sleep(20)
    threading.Thread(target=heartbeat,daemon=True).start()
    def wait_result(path):
        while not path.exists():
            if time.time()>deadline:raise TimeoutError('Campaign prerequisite deadline')
            process_file=path.parent.with_name(path.parent.name+'-process.json')
            if process_file.exists():
                pid=json.loads(process_file.read_text())['pid']
                if not Path('/proc/'+str(pid)).exists():
                    raise RuntimeError('Prerequisite exited without a result: '+str(path))
            time.sleep(5)
        return json.loads(path.read_text())
    def background(module,args,label):
        process_file=a.output/(label+'-process.json')
        if process_file.exists():
            pid=json.loads(process_file.read_text())['pid'];audit_output=Path(args[args.index('--output')+1])
            if process_matches(pid,module,audit_output):
                adopted=ExistingAudit(pid,module,audit_output);background_processes.append(adopted)
                write_state(event/(label+'-adopted.json'),{'pid':pid,'module':module,'status':'ADOPTED_LIVE_AUDIT'})
                return adopted
        audit_output=Path(args[args.index('--output')+1])
        write_state(event/(label+'-partial-copies.json'),{'archived_incomplete_copies':archive_partial_audit(audit_output)})
        log_path=a.output/(label+'.log')
        if log_path.exists():(event/(label+'-previous.log')).write_bytes(log_path.read_bytes())
        log=log_path.open('w')
        env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12')
        proc=subprocess.Popen([os.sys.executable,'-u','-m',module,*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();background_processes.append(proc);write_state(a.output/(label+'-process.json'),{'pid':proc.pid,'started_unix':time.time()});return proc
    def verify_sources():
        for name,sha in plan['source_sha256'].items():
            if file_sha256(name)!=sha:raise AssertionError('Frozen campaign source changed: '+name)
    def native(name,module,args,half,*,evaluation=False):
        verify_sources()
        existing=root/'runs'/name/'report.json'
        closure=root/'snapshots'/name/'closure.json'
        if existing.exists():
            if not a.resume:raise FileExistsError('Use explicit --resume for a completed stage')
            launch=json.loads((closure.parent/'launch.json').read_text())
            if not closure.exists() or json.loads(closure.read_text())['exit_code']!=0 or launch['arguments']!=args:
                raise ValueError('Existing stage is not a matching successful receipt')
            for source,sha in plan['source_sha256'].items():
                if source.endswith('.py') and source in launch['source_sha256'] and launch['source_sha256'][source]!=sha:
                    raise AssertionError('Existing stage used different frozen sources')
            report=json.loads(existing.read_text())
            state_update(name,{'status':'REUSED_COMPLETE_STAGE','result':report['status']})
            return report
        limit=deadline if evaluation else training_cutoff
        if time.time()+120>limit:raise TimeoutError('No phase budget remains')
        cores=list(range(half*32,(half+1)*32));cores=cores[::4] if evaluation else cores
        env=dict(os.environ,RESEARCH_RUN=name,RESEARCH_ENTRY_MODULE=module,
                 RESEARCH_DDP_RANKS=str(len(cores)),NEURON_RT_VISIBLE_CORES=','.join(map(str,cores)),
                 RESEARCH_TIMEOUT_SECONDS=str(min(5400,int(limit-time.time()))))
        state_update(name,{'status':'RUNNING','started_unix':time.time(),'half':half,'module':module})
        log=(a.output/(name+'.log')).open('w')
        result=subprocess.run([os.sys.executable,'-u','-m','scripts.supervise',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close()
        if result.returncode:
            state_update(name,{'status':'FAILED','exit_code':result.returncode});raise RuntimeError('Native phase failed: '+name)
        report=json.loads((root/'runs'/name/'report.json').read_text())
        state_update(name,{'status':'COMPLETE','ended_unix':time.time(),'result':report['status']});return report
    def paired_jobs(items,worker):
        pending=queue.Queue()
        for item in items:pending.put(item)
        stopped=threading.Event()
        def consume(half):
            results=[]
            while not stopped.is_set():
                try:item=pending.get_nowait()
                except queue.Empty:break
                try:results.append(worker(item,half))
                except Exception:
                    stopped.set();raise
            return results
        with cf.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(consume,h) for h in [0,1]];results=[]
            for future in cf.as_completed(futures):results.extend(future.result())
            return results
    try:
        monitor_log=(event/'monitor.log').open('w')
        monitor=subprocess.Popen([os.sys.executable,'-u','-m','scripts.monitor_variable_depth',
            '--campaign',str(a.output),'--pid',str(os.getpid())],stdout=monitor_log,stderr=subprocess.STDOUT,start_new_session=True)
        monitor_log.close();write_state(event/'monitor-process.json',{'pid':monitor.pid})
        qual=wait_result(a.qualification/'report.json')
        if qual['status']!='PASS_VARIABLE_DEPTH_QUALIFICATION':raise ValueError('Native qualification failed')
        fresh=wait_result(a.fresh_panel/'manifest.json')
        if fresh['status']!='PASS_FRESH_NATURAL_DEPTH_PANEL':raise ValueError('Fresh panel failed')
        subprocess.run(['docker','update','--memory',str(320*1024**3),'autonomous-compute-neuron233'],check=True,capture_output=True)
        packed_dir=root/'natural-append-3250m/packed'
        prefix=np.load(packed_dir/'prefix_counts.npy',mmap_mode='r')
        def count(update):return int(prefix[update*256].sum())
        source=f'{croot}/natural-append-3250m/source.json';packed=f'{croot}/natural-append-3250m/packed'
        dev=f'{croot}/{a.fresh_panel.relative_to(root)}/dev.json'
        c=PreNormRecurrenceConfig();step_flops={d:c.__class__(repetitions=d).flops(1024,1024,256)['training_matmul_flops'] for d in [1,2]}
        rates={v['depth_policy']:v['warm_global_real_targets_per_second'] for v in qual['full_size_reports'].values()}
        slow=min(rates.values());budget_targets=12*count(1152)+9*count(3840)+3*(count(6496)-count(3456))
        # Target work by arm, two halves, 30% margin plus an hour for setup/readout.
        core_seconds=sum((4*count(1152)+3*count(3840))/rates[arm] for arm in ['ordinary','fixed','mixture'])
        core_seconds+=3*(count(6496)-count(3456))/rates['ordinary']
        predicted=core_seconds/2*1.3+60*60
        # Recovery validates the saved membership; a new helper is not a new
        # version of an already frozen scientific source.
        frozen_files=list(saved_plan['source_sha256'])
        plan={'identity':'variable-depth-v1','started_unix':started,'deadline_unix':deadline,'training_cutoff_unix':training_cutoff,
             'seeds':SEEDS,'data_seed':23,'fresh_manifest_sha256':file_sha256(a.fresh_panel/'manifest.json'),
             'dev_sha256':fresh['dev_sha256'],'test_sha256':fresh['test_sha256'],
             'training_corpus_sha256':file_sha256(root/'natural-append-3250m/source.json'),
             'lr_candidates':[.00015,.0003,.0006,.001],'tuning_updates':1152,
             'optimizer':{'betas':[.9,.95],'eps':1e-8,'matrix_weight_decay':.1,'embedding_norm_weight_decay':0.,'primary_warmup_updates':96,'tuning_warmup_updates':29},
             'nominal_training_flops_per_step':step_flops,'primary_targets':count(3840),'compute_control_targets':count(6496),
             'budget':{'measured_targets_per_second_per_half_by_arm':rates,'slow_measured_targets_per_second_per_half':slow,'primary_target_work':budget_targets,
                       'conservative_predicted_seconds_including_30_percent_and_one_hour':predicted},
             'source_sha256':{f:file_sha256(f) for f in frozen_files}}
        if a.resume and (a.output/'plan.json').exists():
            old_plan=json.loads((a.output/'plan.json').read_text())
            if old_plan['source_sha256']!=plan['source_sha256'] or old_plan['fresh_manifest_sha256']!=plan['fresh_manifest_sha256']:
                raise AssertionError('Resume changed frozen source or panel')
            plan=old_plan
        write_state(a.output/'plan.json',plan)
        public=Path('results/variable_depth_20261023');public.mkdir(exist_ok=True)
        write_state(public/'plan.json',plan);write_state(public/'data-gate.json',fresh)
        base=['--corpus',source,'--packed-corpus',packed,'--dev-corpus',dev,'--context','1024','--batch','4',
              '--accumulation','2','--architecture','preln','--precision','bf16','--optimizer','fused','--recipe','pretraining',
              '--data-seed','23','--reference-threads','4','--warmup','4','--save','--checkpoint-every','1024']
        panel_args=['--corpus',dev,'--split','dev','--cohort','curve']
        def evaluate(model,depth,half,split='dev',cohort='curve'):
            name=model['identity']+f'-{split}-r{depth}'
            args=['--checkpoint',model['checkpoint'],'--corpus',dev if split=='dev' else f'{croot}/{a.fresh_panel.relative_to(root)}/test-sealed.json',
                   '--repetitions',str(depth),'--split',split,'--cohort','all' if split=='test' else cohort]
            if split=='test':args+=['--test-release',f'{croot}/{a.output.relative_to(root)}/test-release.json']
            return native(name,'scripts.evaluate_depth_curve',args,half,evaluation=True)
        def tune(item,half):
            arm,lr=item;name=f'vdepth-v1-tune-{arm}-lr{int(lr*1e6)}'
            args=base+['--purpose','tuning','--seed','11','--depth-policy',arm,'--lr',str(lr),
                       '--steps','1152','--lr-warmup-tokens',str(count(29)),
                       '--decay-start',str(count(1037)),'--decay-end',str(count(1152))]
            report=native(name,'scripts.train_variable_depth',args,half)
            model={'identity':name,'checkpoint':f'{croot}/runs/{name}/checkpoint.pt'}
            dev_report=evaluate(model,1 if arm=='ordinary' else 2,half,cohort='tune')
            return {'arm':arm,'lr':lr,'run':name,'metrics':dev_report['metrics'],'initial_model_digest':report['initial_model_digest']}
        state_update('tuning',{'status':'RUNNING'})
        tunes=paired_jobs([(arm,lr) for lr in plan['lr_candidates'] for arm in ['ordinary','fixed','mixture']],tune)
        selected={arm:min((v for v in tunes if v['arm']==arm),key=lambda v:v['metrics']['weighted_nll']) for arm in ['ordinary','fixed','mixture']}
        tuning_path=a.output/'tuning.json'
        if tuning_path.exists():
            tuning_record=json.loads(tuning_path.read_text())
            for arm,v in tuning_record['selected'].items():
                if not any(t['run']==v['run'] and t['lr']==v['lr'] and t['metrics']==v['metrics'] for t in tunes):
                    raise AssertionError('Previously selected LR trial changed')
            selected=tuning_record['selected']
        else:
            tuning_record={'trials':tunes,'selected':selected};write_state(tuning_path,tuning_record)
        write_state(public/'tuning.json',tuning_record)
        if any(v['metrics']['weighted_nll']>=5 for arm,v in selected.items() if arm!='ordinary'):
            raise ValueError('Declared recurrent learning gate failed; no main fits')
        state_update('tuning',{'status':'COMPLETE','selected_lrs':{arm:v['lr'] for arm,v in selected.items()}})
        inventory=[]
        for seed in SEEDS:
            for arm in ['ordinary','fixed','mixture']:
                for endpoint,update in [('1B',3840)]+([('matched',6496)] if arm=='ordinary' else []):
                    identity=f'vdepth-v1-{arm}-s{seed}-u{update}'
                    inventory.append({'identity':identity,'arm':arm,'seed':seed,'endpoint':endpoint,'parameters':c.parameter_count(),
                       'checkpoint':f'{croot}/runs/{identity}/checkpoint.pt','allowed_depths':[1] if arm=='ordinary' else [1,2,3,4]})
        write_state(a.output/'core-models.json',inventory)
        audit=background('scripts.audit_variable_depth',['--root',str(root),'--models',str(a.output/'core-models.json'),
             '--corpus',str(a.fresh_panel/'dev.json'),'--output',str(a.output/'independent-cpu'),'--deadline-unix',str(deadline-300)],'independent-cpu')
        all_initial={};stable_parents={}
        def fit(item,half):
            arm,seed=item;previous=None;prior=0;main_report=None
            common=base+['--purpose','scientific','--seed',str(seed),'--depth-policy',arm,'--lr',str(selected[arm]['lr']),
                          '--lr-warmup-tokens',str(count(96)),'--decay-start',str(count(3456)),'--decay-end',str(count(3840))]
            for update in [384,1152,3456,3840]:
                name=f'vdepth-v1-{arm}-s{seed}-u{update}'
                args=common+['--steps',str(update-prior)]
                if previous:args+=['--resume',previous]
                report=native(name,'scripts.train_variable_depth',args,half)
                if report['total_real_targets']!=count(update) or not report['all_replica_model_optimizer_states_equal']:
                    raise AssertionError('Source exposure or replica state differs')
                if previous and not report['recovered_model_optimizer_digest']:raise AssertionError('Recovery missing')
                with lock:
                    key=(seed,update);all_initial.setdefault(key,set()).add(report['initial_model_digest'])
                    if len(all_initial[key])!=1:raise AssertionError('Paired initialization differs')
                previous=f'{croot}/runs/{name}/checkpoint.pt';prior=update
                if update==3456:stable_parents[arm,seed]=previous
                if update in [384,1152,3840]:
                    model={'identity':name,'checkpoint':previous};evaluate(model,1 if arm=='ordinary' else 2,half)
                main_report=report
            if arm=='ordinary':
                previous=stable_parents[arm,seed];prior=3456
                extended=[v for v in common] # Replace only future decay boundaries, preserve completed history.
                extended[extended.index('--decay-start')+1]=str(count(5846));extended[extended.index('--decay-end')+1]=str(count(6496))
                for update in [5846,6496]:
                    name=f'vdepth-v1-{arm}-s{seed}-u{update}'
                    report=native(name,'scripts.train_variable_depth',extended+['--steps',str(update-prior),
                        '--resume',previous,'--extend-stable-schedule'],half)
                    previous=f'{croot}/runs/{name}/checkpoint.pt';prior=update
                evaluate({'identity':name,'checkpoint':previous},1,half)
            return {'arm':arm,'seed':seed,'main_state_digest':main_report['state_digest']}
        state_update('primary_fits',{'status':'RUNNING'})
        fit_tasks=[(arm,seed) for seed in SEEDS for arm in ['ordinary','fixed','mixture']]
        fit_tasks.sort(key=lambda item:-(count(3840)+(count(6496)-count(3456) if item[0]=='ordinary' else 0))/rates[item[0]])
        paired_jobs(fit_tasks,fit)
        state_update('primary_fits',{'status':'COMPLETE'})
        if audit.poll() is not None and audit.returncode:raise RuntimeError('Independent audit failed')
        remaining_long_targets=3*(count(11520)-count(3456))
        long_estimate=(count(11520)-count(3456))*sum(1/rates[arm] for arm in ['ordinary','fixed','mixture'])/2*1.3+20*60
        long_models=[]
        if estimate['previously_launched_longer_stage'] or time.time()+long_estimate<training_cutoff:
            state_update('longer_exposure',{'status':'RUNNING','predicted_seconds':long_estimate})
            def longer(arm,half):
                previous=stable_parents[arm,SEEDS[0]];prior=3456
                for update in [7680,10368,11520]:
                    name=f'vdepth-v1-long-{arm}-s{SEEDS[0]}-u{update}'
                    args=base+['--purpose','scientific','--seed',str(SEEDS[0]),'--depth-policy',arm,
                        '--lr',str(selected[arm]['lr']),'--lr-warmup-tokens',str(count(96)),
                        '--decay-start',str(count(10368)),'--decay-end',str(count(11520)),
                        '--steps',str(update-prior),'--resume',previous,'--extend-stable-schedule']
                    report=native(name,'scripts.train_variable_depth',args,half)
                    previous=f'{croot}/runs/{name}/checkpoint.pt';prior=update
                    evaluate({'identity':name,'checkpoint':previous},1 if arm=='ordinary' else 2,half)
                return {'identity':name,'arm':arm,'seed':SEEDS[0],'endpoint':'3B','parameters':c.parameter_count(),
                        'checkpoint':previous,'allowed_depths':[1] if arm=='ordinary' else [1,2,3,4]}
            long_models=paired_jobs(sorted(['ordinary','fixed','mixture'],key=lambda arm:rates[arm]),longer);inventory+=long_models
            state_update('longer_exposure',{'status':'COMPLETE'})
        else:state_update('longer_exposure',{'status':'UNEXECUTED_TIME_GATE','predicted_seconds':long_estimate})
        for model in inventory:
            report=json.loads((root/'runs'/model['identity']/'report.json').read_text())
            model['training_matmul_flops']=report['total_training_matmul_flops']
            model['training_real_targets']=report['total_real_targets']
            model['checkpoint_sha256']=file_sha256(root/'runs'/model['identity']/'checkpoint.pt')
        for seed in SEEDS:
            paired={m['arm']:m for m in inventory if m['seed']==seed and m['endpoint']=='1B'}
            if len(paired)!=3 or any(m['parameters']!=c.parameter_count() for m in paired.values()):
                raise AssertionError('Missing or parameter-mismatched paired arms')
            if paired['fixed']['training_matmul_flops']!=paired['mixture']['training_matmul_flops']:
                raise AssertionError('Fixed/mixture update work differs')
            ordinary=next(m for m in inventory if m['seed']==seed and m['endpoint']=='matched')
            shortfall=1-ordinary['training_matmul_flops']/paired['mixture']['training_matmul_flops']
            if not 0<=shortfall<.00014:raise AssertionError('Matched-update-work endpoint differs')
        write_state(a.output/'models.json',inventory)
        # Complete the predeclared DEV matrix; existing canonical cells are reused exactly.
        cells=[(m,d) for m in inventory for d in m['allowed_depths'] if not (root/'runs'/f"{m['identity']}-dev-r{d}"/'report.json').exists()]
        paired_jobs(cells,lambda cell,half:evaluate(*cell,half))
        subprocess.run([os.sys.executable,'-m','scripts.analyze_variable_depth','--root',str(root),'--inventory',str(a.output/'models.json'),
             '--split','dev','--output',str(public/'dev')],check=True)
        while audit.poll() is None:
            if time.time()>deadline-1800:raise TimeoutError('Independent audit did not complete before TEST window')
            time.sleep(5)
        if audit.returncode:raise RuntimeError('Independent audit failed')
        write_state(public/'independent-cpu-verification.json',json.loads((a.output/'independent-cpu/report.json').read_text()))
        if long_models:
            write_state(a.output/'long-models.json',long_models)
            second=background('scripts.audit_variable_depth',['--root',str(root),'--models',str(a.output/'long-models.json'),
                '--corpus',str(a.fresh_panel/'dev.json'),'--output',str(a.output/'long-independent-cpu'),'--deadline-unix',str(deadline-120)],'long-independent-cpu')
        # No more model/LR/endpoint/depth decisions occur after this release.
        verify_sources()
        release={'status':'FROZEN_VARIABLE_DEPTH_TEST_RELEASE','frozen_unix':time.time(),'models':inventory,
             'corpus_sha256':fresh['test_sha256'],'evaluator_sha256':file_sha256('scripts/evaluate_depth_curve.py'),
             'analysis_sha256':file_sha256('scripts/analyze_variable_depth.py'),'source_sha256':plan['source_sha256']}
        existing_release=a.output/'test-release.json'
        release=preserve_test_release(release,json.loads(existing_release.read_text()) if existing_release.exists() else None)
        write_state(a.output/'test-release.json',release);write_state(public/'test-release.json',release)
        state_update('frozen_test',{'status':'RUNNING','models':len(inventory)})
        paired_jobs([(m,d) for m in inventory for d in m['allowed_depths']],lambda cell,half:evaluate(*cell,half,split='test'))
        subprocess.run([os.sys.executable,'-m','scripts.analyze_variable_depth','--root',str(root),'--inventory',str(a.output/'models.json'),
             '--split','test','--output',str(public/'test')],check=True)
        if long_models:
            second.wait(timeout=max(1,deadline-time.time()-30))
            if second.returncode:raise RuntimeError('Longer endpoint audit failed')
            write_state(public/'long-independent-cpu-verification.json',json.loads((a.output/'long-independent-cpu/report.json').read_text()))
        state['status']='COMPLETE_VARIABLE_DEPTH_CAMPAIGN';state_update('frozen_test',{'status':'COMPLETE'})
        write_state(a.output/'report.json',{'status':state['status'],'elapsed_seconds':time.time()-plan['started_unix'],
                    'recovery_execution_seconds':time.time()-started,'recovery_amendment_sha256':file_sha256(event/'amendment.json'),'models':len(inventory),
                    'longer_exposure_executed':bool(long_models)})
    except Exception as error:
        cleanup_errors=[]
        for process in background_processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except (TimeoutError,subprocess.TimeoutExpired):process.kill();process.wait(timeout=5)
            except Exception as cleanup_error:cleanup_errors.append(repr(cleanup_error))
        state['status']='STOPPED_VARIABLE_DEPTH_CAMPAIGN';state_update('failure',{'type':type(error).__name__,'reason':str(error)})
        write_state(a.output/'report.json',{'status':state['status'],'error':str(error),'elapsed_seconds':time.time()-started,
                                         'cleanup_errors':cleanup_errors})
        raise


if __name__=='__main__':main()
