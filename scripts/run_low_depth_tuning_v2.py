"""Finite paired-hardware tuning queue with independent CPU scoring."""
import argparse
import concurrent.futures as cf
import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running,cancel_worker_container
from compute_trainium.owned_native_compact import create_compact_worker,reclaim_host
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def choose_rates(records):
    selected={}
    for arm in ['ordinary','fixed1','poisson2']:
        rows=[r for r in records if r['arm']==arm and r['status']=='COMPLETE_INDEPENDENT_LOW_DEPTH_DEVELOPMENT_SCORE']
        if not rows:raise ValueError('No completed eligible LR for '+arm)
        selected[arm]=min(rows,key=lambda r:(r['metrics']['weighted_nll'],r['peak_lr']))
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    experiment=a.root/'parcae-low-depth-pilot-v1';plan_file=experiment/'tuning-plan-v2.json'
    plan=json.loads(plan_file.read_text());plan_sha=file_sha256(plan_file)
    out=experiment/'tuning-execution-v2';public=Path('results/parcae_low_depth_pilot_20261024/tuning-v2')
    out.mkdir(exist_ok=True);public.mkdir(parents=True,exist_ok=True)
    started_file=out/'execution-start.json'
    if started_file.exists():
        if not a.resume:raise FileExistsError('Existing execution requires explicit --resume')
        execution=json.loads(started_file.read_text())
        if execution['plan_sha256']!=plan_sha:raise ValueError('Resume changed the frozen tuning plan')
    else:
        if a.resume:raise FileNotFoundError('No recorded tuning execution to resume')
        execution={'started_unix':time.time(),'plan_sha256':plan_sha,'deadline_unix':time.time()+plan['seconds_limit']}
        write_state(started_file,execution)
    deadline=execution['deadline_unix']
    if time.time()>min(deadline,plan['latest_start_unix']):raise TimeoutError('Registered tuning execution window expired')
    def frozen():
        if file_sha256(plan_file)!=plan_sha or any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):
            raise AssertionError('Frozen tuning source/plan changed')
        if any(file_sha256(r['path'])!=r['sha256'] for r in plan['qualification_receipts'].values()):
            raise AssertionError('Qualification receipt changed')
    frozen()
    if (out/'report.json').exists():
        old=json.loads((out/'report.json').read_text())
        if old['status']=='COMPLETE_FROZEN_LOW_DEPTH_LR_SCREEN':return
        # The stopped terminal receipt remains; an explicit resume is recorded.
        if not a.resume:raise FileExistsError('Stopped execution needs explicit recovery')
        archive=out/f'stopped-{int(time.time())}.json';write_state(archive,old)
    tasks=queue.Queue()
    for case_id in plan['cases']:tasks.put(case_id)
    lock=threading.Lock();failed=threading.Event();active={};cpu_active={};costs=[];scores=[];trials=[];cpu_costs=[]
    def interrupted(signum,frame):
        failed.set();raise KeyboardInterrupt('Registered tuning execution interrupted by signal '+str(signum))
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    score_pool=cf.ThreadPoolExecutor(4);score_futures=[]

    def native(case_id,half,phase,parent=None):
        frozen();reclaim_host()
        if time.time()>min(deadline,plan['latest_start_unix']):raise TimeoutError('Tuning latest phase start')
        identity=f'low-depth-tune-v2-{case_id}-{phase}';path=a.root/'runs'/identity
        expected_end=20 if phase=='first' else 576
        report_file=path/'report.json'
        if report_file.exists():
            report=json.loads(report_file.read_text())
            if (report['status']!='COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT' or report['plan_sha256']!=plan_sha
                or report['case_id']!=case_id or report['update']!=expected_end or report['qualification_only']):
                raise ValueError('Completed tuning phase identity differs')
            record=next(r for r in report['checkpoint_records'] if r['update']==expected_end)
            if file_sha256(path/f'checkpoint-u{expected_end}.pt')!=record['checkpoint_sha256']:
                raise AssertionError('Completed tuning checkpoint changed')
            return identity,report
        if path.exists():raise RuntimeError('Partial native phase preserved; register a checksum-qualified recovery identity before reuse: '+identity)
        container=create_compact_worker(a.root,identity,half);process=None
        with lock:active[identity]=container
        try:
            limit=min(deadline,time.time()+plan['phase_seconds_limit'])
            env=dict(os.environ,RESEARCH_ROOT=str(a.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                RESEARCH_RUN=identity,RESEARCH_OWNER=identity,RESEARCH_CONTAINER=container,
                RESEARCH_ENTRY_MODULE='scripts.train_low_depth_runtime',RESEARCH_DDP_RANKS='32',
                NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,half*32+32))),
                RESEARCH_TIMEOUT_SECONDS=str(max(1,int(limit-time.time()))))
            command=[os.sys.executable,'-u','-m','scripts.supervise_isolated','--root','/work/instance-nvme',
                '--plan','/work/instance-nvme/parcae-low-depth-pilot-v1/tuning-plan-v2.json','--plan-sha',plan_sha,
                '--case',case_id,'--segment-end',str(expected_end),'--comm-port',str(plan['root_comm_ports'][identity])]
            if parent:
                checkpoint=next(r for r in parent[1]['checkpoint_records'] if r['update']==20)
                command+=['--resume','/work/instance-nvme/runs/'+parent[0]+'/checkpoint-u20.pt',
                          '--resume-sha',checkpoint['checkpoint_sha256']]
            log_file=out/(identity+f'-launch-{int(time.time())}.log')
            with log_file.open('x') as log:process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if failed.is_set() or time.time()>limit:raise RuntimeError('Tuning paired stop/phase deadline')
                time.sleep(5)
            if process.returncode:
                text=(a.root/'logs'/(identity+'.log')).read_text()
                nonfinite=any(s in text for s in ['Nonfinite scientific training loss','Nonfinite full global gradient'])
                if nonfinite:return identity,{'status':'FAILED_NONFINITE_LR_OPPORTUNITY','case_id':case_id,
                    'arm':plan['cases'][case_id]['arm'],'peak_lr':plan['cases'][case_id]['peak_lr']}
                raise RuntimeError('Native tuning phase failed: '+identity)
            report=json.loads(report_file.read_text())
            if (report['status']!='COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT' or report['update']!=expected_end
                or report['qualification_only'] or report['case_id']!=case_id or report['plan_sha256']!=plan_sha):
                raise AssertionError('Native tuning completion differs from frozen case')
            if parent:
                original=next(r for r in parent[1]['checkpoint_records'] if r['update']==20)
                if not report['complete_state_on_load_exact'] or report['recovered_state_digest']!=original['full_state_digest']:
                    raise AssertionError('Scientific tuning recovery changed complete state')
            return identity,report
        finally:
            cancel_worker_container(container,identity,'tuning_native_phase_complete')
            if process and process.poll() is None:
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.terminate()
            closure=a.root/'snapshots'/identity/'closure.json'
            with lock:
                active.pop(identity,None)
                costs.append({'identity':identity,**json.loads(closure.read_text())} if closure.exists() else {'identity':identity,'closure_missing':True})

    def score(case_id,identity,report):
        begin=time.time();process=None;target=out/'cpu-scores'/case_id
        try:
            frozen()
            checkpoint=next(r for r in report['checkpoint_records'] if r['update']==576)
            if (target/'report.json').exists():
                row=json.loads((target/'report.json').read_text())
                if row['checkpoint_sha256']!=checkpoint['checkpoint_sha256']:raise AssertionError('Resumed score checkpoint changed')
            else:
                if target.exists():raise RuntimeError('Partial CPU score preserved; register an explicit score recovery')
                target.parent.mkdir(exist_ok=True)
                env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12')
                command=[os.sys.executable,'-u','-m','scripts.evaluate_low_depth_cpu',
                    '--root',str(a.root),'--plan',str(plan_file),'--plan-sha',plan_sha,
                    '--checkpoint',str(a.root/'runs'/identity/'checkpoint-u576.pt'),
                    '--checkpoint-sha',checkpoint['checkpoint_sha256'],'--case',case_id,
                    '--cohort','tune','--depth',str(plan['cases'][case_id]['primary_inference_depth']),
                    '--output',str(target),'--upstream',str(a.root/'upstream-parcae-69284c1'),
                    '--seconds-limit',str(plan['cpu_score_seconds_limit'])]
                with (out/(case_id+f'-cpu-{int(time.time())}.log')).open('x') as log:
                    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with lock:cpu_active[case_id]=process
                limit=min(deadline,begin+plan['cpu_score_seconds_limit'])
                while process.poll() is None:
                    if failed.is_set() or time.time()>limit:raise RuntimeError('CPU tuning score paired stop/deadline')
                    time.sleep(5)
                if process.returncode:raise RuntimeError('Independent tuning CPU score failed: '+case_id)
                row=json.loads((target/'report.json').read_text())
            if row['status']!='COMPLETE_INDEPENDENT_LOW_DEPTH_DEVELOPMENT_SCORE':raise AssertionError('CPU score incomplete')
            row['peak_lr']=plan['cases'][case_id]['peak_lr']
            with lock:scores.append(row);write_state(public/'scores-progress.json',{'records':scores})
            return row
        except BaseException:failed.set();raise
        finally:
            if process and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            with lock:
                cpu_active.pop(case_id,None)
                cpu_costs.append({'case_id':case_id,'started_unix':begin,'finished_unix':time.time(),
                    'wall_seconds':time.time()-begin,'assigned_chips':0,'exit_code':process.returncode if process else None,
                    'reused_completed_score':process is None})

    def worker(half):
        try:
            while not failed.is_set():
                try:case_id=tasks.get_nowait()
                except queue.Empty:return
                first=native(case_id,half,'first')
                final=native(case_id,half,'fit',first) if first[1]['status']!='FAILED_NONFINITE_LR_OPPORTUNITY' else first
                row={'case_id':case_id,'arm':plan['cases'][case_id]['arm'],'peak_lr':plan['cases'][case_id]['peak_lr'],
                    'status':final[1]['status'],'final_identity':final[0]}
                with lock:trials.append(row);write_state(public/'training-progress.json',{'records':trials})
                if final[1]['status']=='COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT':
                    future=score_pool.submit(score,case_id,*final)
                    with lock:score_futures.append(future)
        except BaseException:failed.set();raise

    try:
        while managed_running():
            if time.time()>deadline:raise TimeoutError('Tuning admission deadline')
            write_state(public/'status.json',{'status':'WAITING_FOR_QUALIFICATION_OWNERS_TO_RELEASE','assigned_chips':0,'observed_unix':time.time()})
            time.sleep(5)
        write_state(public/'status.json',{'status':'RUNNING_FROZEN_THREE_ARM_LR_SCREEN','maximum_chips':16,'started_unix':execution['started_unix']})
        with cf.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(worker,half) for half in [0,1]]
            for future in cf.as_completed(futures):future.result()
        for future in score_futures:future.result(timeout=max(1,deadline-time.time()))
        frozen()
        if time.time()>deadline:raise TimeoutError('Tuning complete execution budget expired')
        selected=choose_rates(scores)
        result={'status':'COMPLETE_FROZEN_LOW_DEPTH_LR_SCREEN','selected':selected,'scores':scores,'trials':trials}
    except BaseException as e:
        failed.set();result={'status':'STOPPED_FROZEN_LOW_DEPTH_LR_SCREEN','error':str(e),'scores':scores,'trials':trials};raise
    finally:
        failed.set()
        for identity,container in list(active.items()):
            try:cancel_worker_container(container,identity,'tuning_execution_stop')
            except BaseException:pass
        for process in list(cpu_active.values()):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        score_pool.shutdown(wait=True,cancel_futures=True)
        result.update(plan_sha256=plan_sha,wall_seconds=time.time()-execution['started_unix'],test_accessed=False,
                      new_terminal_attempt=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0,'observed_unix':time.time()})
        write_state(public/'execution-costs.json',{'native_attempts':costs,'cpu_scores':cpu_costs,
            'wall_seconds':time.time()-execution['started_unix'],'scope':'All launched scientific LR paths and failures, including initial20-update segments, fresh-process continuation, compilation, checkpointing and CPU scoring. Reused completed stages are not rerun. Concurrent intervals are not sequential reservation time.'})


if __name__=='__main__':main()
