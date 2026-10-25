"""Execution-only recovery of the unchanged small-core scientific pilot."""
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
from scripts.run_low_depth_gradient_validation import register as register_gradient
from compute_trainium.low_depth_recovery import gradient_identity


def checkpoint_row(report,update):
    return next(r for r in report['checkpoint_records'] if r['update']==update)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    experiment=a.root/'parcae-low-depth-pilot-v1';plan_file=experiment/'main-plan-recovery-v2.json'
    plan=json.loads(plan_file.read_text());plan_sha=file_sha256(plan_file)
    out=experiment/'main-execution-recovery-v2';public=Path('results/parcae_low_depth_pilot_20261024/main-recovery-v2')
    out.mkdir(exist_ok=True);started_file=out/'execution-start.json'
    if started_file.exists():
        if not a.resume:raise FileExistsError('Explicit --resume required')
        execution=json.loads(started_file.read_text())
        if execution['plan_sha256']!=plan_sha:raise ValueError('Main recovery plan changed')
    else:
        if a.resume:raise ValueError('No main execution to recover')
        execution={'started_unix':time.time(),'deadline_unix':time.time()+plan['seconds_limit'],'plan_sha256':plan_sha}
        write_state(started_file,execution)
    deadline=execution['deadline_unix']
    if time.time()>deadline:raise TimeoutError('Original registered main window expired')
    if (out/'report.json').exists():
        previous=json.loads((out/'report.json').read_text())
        if previous['status']=='COMPLETE_FROZEN_LOW_DEPTH_MAIN_PILOT':return
        if not a.resume:raise FileExistsError('Stopped main execution needs explicit recovery')
        write_state(out/f'stopped-{int(time.time())}.json',previous)
    def frozen():
        if file_sha256(plan_file)!=plan_sha or any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):
            raise AssertionError('Frozen main source/plan changed')
        if any(file_sha256(r['path'])!=r['sha256'] for r in plan['launch_gate_receipts'].values()):
            raise AssertionError('Main numerical/tuning launch receipt changed')
    lock=threading.Lock();failed=threading.Event();halves=queue.Queue()
    for half in [0,1]:halves.put(half)
    active={};cpu_active={};costs=[];cpu_costs=[];native_records=[];gradient_records=[];scores=[]
    def interrupted(signum,frame):
        failed.set();raise KeyboardInterrupt('Registered main execution interrupted by signal '+str(signum))
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    cpu_pool=cf.ThreadPoolExecutor(3)
    def stopped():
        if failed.is_set() or time.time()>deadline:raise RuntimeError('Main paired stop/deadline')
    def acquire():
        while True:
            stopped()
            try:return halves.get(timeout=5)
            except queue.Empty:pass
    def native(identity,module,args,*,limit_seconds):
        frozen();stopped();half=acquire();container=None;process=None
        try:
            reclaim_host();container=create_compact_worker(a.root,identity,half)
            with lock:active[identity]=container
            limit=min(deadline,time.time()+limit_seconds)
            env=dict(os.environ,RESEARCH_ROOT=str(a.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                RESEARCH_RUN=identity,RESEARCH_OWNER=identity,RESEARCH_CONTAINER=container,
                RESEARCH_ENTRY_MODULE=module,RESEARCH_DDP_RANKS='32',
                NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,half*32+32))),
                RESEARCH_TIMEOUT_SECONDS=str(max(1,int(limit-time.time()))))
            command=[os.sys.executable,'-u','-m','scripts.supervise_isolated',*args]
            with (out/(identity+f'-{int(time.time())}.log')).open('x') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                stopped()
                if time.time()>limit:raise TimeoutError('Main native phase deadline: '+identity)
                time.sleep(5)
            if process.returncode:raise RuntimeError('Main native phase failed: '+identity)
            return json.loads((a.root/'runs'/identity/'report.json').read_text())
        except BaseException:failed.set();raise
        finally:
            if container:
                cancel_worker_container(container,identity,'main_owned_phase_closed')
                if process and process.poll() is None:
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.terminate()
                closure=a.root/'snapshots'/identity/'closure.json'
                with lock:
                    active.pop(identity,None)
                    costs.append({'identity':identity,**json.loads(closure.read_text())} if closure.exists() else {'identity':identity,'closure_missing':True})
            halves.put(half)
    def cpu_job(key,module,args,report_file,expected_status,seconds):
        frozen();stopped();begin=time.time();process=None
        try:
            if report_file.exists():
                row=json.loads(report_file.read_text())
            else:
                if report_file.parent.exists():raise RuntimeError('Preserve partial CPU phase; explicit recovery identity required: '+key)
                env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12')
                with (out/(key+f'-{int(time.time())}.log')).open('x') as log:
                    process=subprocess.Popen([os.sys.executable,'-u','-m',module,*args],env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with lock:cpu_active[key]=process
                while process.poll() is None:
                    stopped()
                    if time.time()>begin+seconds:raise TimeoutError('Main CPU phase cap: '+key)
                    time.sleep(5)
                if process.returncode:raise RuntimeError('Main independent CPU phase failed: '+key)
                row=json.loads(report_file.read_text())
            if row['status']!=expected_status:raise AssertionError('CPU result failed qualification: '+key)
            return row
        except BaseException:failed.set();raise
        finally:
            if process and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            with lock:
                cpu_active.pop(key,None);cpu_costs.append({'identity':key,'assigned_chips':0,'wall_seconds':time.time()-begin,
                    'started_unix':begin,'finished_unix':time.time(),'reused':process is None,
                    'exit_code':process.returncode if process else None})
    def fit(arm,phase,parent=None):
        case_id=arm+('-equal-s1231' if phase=='equal' else '-main-s1231');case=plan['cases'][case_id]
        end=1920 if phase in ['first','equal'] else case['total_updates']
        identity=f'low-depth-main-r2-{arm}-{phase}';path=a.root/'runs'/identity
        if (path/'report.json').exists():
            report=json.loads((path/'report.json').read_text())
        else:
            if path.exists():raise RuntimeError('Preserve partial native scientific fit; checksum-qualified recovery registration required: '+identity)
            args=['--root','/work/instance-nvme','--plan','/work/instance-nvme/parcae-low-depth-pilot-v1/main-plan-recovery-v2.json',
                '--plan-sha',plan_sha,'--case',case_id,'--segment-end',str(end),'--comm-port',str(plan['root_comm_ports'][identity])]
            if parent:
                parent_update=960 if phase=='equal' else 1920;row=checkpoint_row(parent[1],parent_update)
                args+=['--resume','/work/instance-nvme/runs/'+parent[0]+f'/checkpoint-u{parent_update}.pt',
                       '--resume-sha',row['checkpoint_sha256']]
            report=native(identity,'scripts.train_low_depth_runtime',args,limit_seconds=plan['native_phase_seconds_limit'])
        if (report['status']!='COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT' or report['plan_sha256']!=plan_sha
            or report['case_id']!=case_id or report['update']!=end or report['qualification_only']):
            raise AssertionError('Main completion identity changed')
        for row in report['checkpoint_records']:
            if file_sha256(path/f'checkpoint-u{row["update"]}.pt')!=row['checkpoint_sha256'] or not row['all_complete_replica_states_equal']:
                raise AssertionError('Main checkpoint/replica verification failed')
        if parent:
            row=checkpoint_row(parent[1],960 if phase=='equal' else 1920)
            if not report['complete_state_on_load_exact'] or report['recovered_state_digest']!=row['full_state_digest']:
                raise AssertionError('Main full-state recovery differs')
        if phase!='first' and (report['real_targets']!=case['end_targets'] or report['executed_matmuls']!=case['training_matmuls']):
            raise AssertionError('Main terminal source/work counters differ')
        with lock:
            native_records.append({'identity':identity,**{k:v for k,v in report.items() if k!='progress'}})
            write_state(public/'training-progress.json',{'records':native_records})
        return identity,report
    def score(arm,label,fit_result,update,depth):
        identity,report=fit_result;row=checkpoint_row(report,update);case_id=report['case_id']
        key=f'{arm}-{label}-d{depth}';target=out/'scores'/key;target.parent.mkdir(exist_ok=True)
        args=['--root',str(a.root),'--plan',str(plan_file),'--plan-sha',plan_sha,
            '--checkpoint',str(a.root/'runs'/identity/f'checkpoint-u{update}.pt'),
            '--checkpoint-sha',row['checkpoint_sha256'],'--case',case_id,'--cohort','curve',
            '--depth',str(depth),'--output',str(target),'--upstream',str(a.root/'upstream-parcae-69284c1'),
            '--seconds-limit',str(plan['cpu_score_seconds_limit'])]
        result=cpu_job(key,'scripts.evaluate_low_depth_cpu',args,target/'report.json',
            'COMPLETE_INDEPENDENT_LOW_DEPTH_DEVELOPMENT_SCORE',plan['cpu_score_seconds_limit'])
        if result['checkpoint_sha256']!=row['checkpoint_sha256']:raise AssertionError('Reused score checkpoint differs')
        result['readout']=label
        with lock:scores.append(result);write_state(public/'scores-progress.json',{'records':scores})
        return result
    def gradient(arm,label,fit_result,update):
        identity,report=fit_result;row=checkpoint_row(report,update)
        if label=='stable1B' and arm in ['fixed1','poisson2']:
            gate_file=Path('results/low-depth-recovery-v2-gradient-gates/report.json')
            while not gate_file.exists():frozen();stopped();time.sleep(5)
            gates=json.loads(gate_file.read_text())
            if gates['status']!='PASS_LOW_DEPTH_GRADIENT_VALIDATION':raise RuntimeError('Recovery first1B numerical prerequisite failed')
            proof=next(r for r in gates['records'] if r['independent_cpu']['arm']==arm)
            audit=proof['independent_cpu'];origin=report['adopted_from']['checkpoints'][str(update)]
            if audit['checkpoint_sha256']!=origin['original_sha256'] or proof['native']['full_state_digest']!=row['full_state_digest']:
                raise AssertionError('Adopted checkpoint and exact predecessor numerical evidence differ')
            with lock:
                gradient_records.append({'arm':arm,'readout':label,'capture_identity':proof['native']['identity'] if 'identity' in proof['native'] else 'low-depth-recovery-v2-gradient-gates-'+arm,
                    'independent_cpu':audit,'adopted_checkpoint_sha256':row['checkpoint_sha256'],
                    'shared_complete_state_digest':row['full_state_digest'],'diagnostic_matmuls':proof['native']['neural_matmuls'],
                    'diagnostic_real_targets':proof['native']['real_targets'],
                    'scope':'Same exact full model/optimizer/RNG/source/depth state;only execution-plan signature rebound,with original numerical receipt retained unchanged.'})
                write_state(public/'gradient-progress.json',{'records':gradient_records})
            return audit
        execution_id=gradient_identity(arm,label);probe_dir=a.root/execution_id
        point=arm;point_file=out/(execution_id+'-points.json')
        manifest={point:{'case_id':report['case_id'],'checkpoint':f'runs/{identity}/checkpoint-u{update}.pt',
                         'checkpoint_sha256':row['checkpoint_sha256']}}
        probe_file=probe_dir/'plan.json'
        if not probe_file.exists():
            write_state(point_file,manifest);probe=register_gradient(a.root,execution_id,plan_file,point_file)
        else:
            probe=json.loads(probe_file.read_text())
            if probe['points']!=manifest:raise AssertionError('Existing main gradient point differs')
        probe_sha=file_sha256(probe_file);capture_id=execution_id+'-'+point;capture_dir=a.root/'runs'/capture_id
        if (capture_dir/'report.json').exists():capture=json.loads((capture_dir/'report.json').read_text())
        else:
            if capture_dir.exists():raise RuntimeError('Partial gradient capture needs explicit recovery')
            capture=native(capture_id,'scripts.capture_low_depth_gradient',
                ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+execution_id+'/plan.json',
                 '--plan-sha',probe_sha,'--point',point,'--comm-port',str(probe['root_comm_ports'][capture_id])],
                limit_seconds=probe['phase_seconds_limit'])
        if capture['status']!='CAPTURED_ZERO_UPDATE_LOW_DEPTH_GLOBAL_GRADIENT' or capture['probe_plan_sha256']!=probe_sha:
            raise AssertionError('Main gradient capture identity differs')
        target=probe_dir/'cpu';args=['--root',str(a.root),'--plan',str(probe_file),'--plan-sha',probe_sha,
            '--point',point,'--capture',str(capture_dir),'--output',str(target),
            '--upstream',str(a.root/'upstream-parcae-69284c1'),'--seconds-limit',str(probe['cpu_seconds_limit'])]
        audited=cpu_pool.submit(cpu_job,execution_id,'scripts.audit_low_depth_gradient',args,target/'report.json',
            'PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT',probe['cpu_seconds_limit']).result()
        with lock:
            gradient_records.append({'arm':arm,'readout':label,'capture_identity':capture_id,'independent_cpu':audited,
                'diagnostic_matmuls':capture['neural_matmuls'],'diagnostic_real_targets':capture['real_targets']})
            write_state(public/'gradient-progress.json',{'records':gradient_records})
        return audited
    def arm_worker(arm):
        try:
            depth=plan['cases'][arm+'-main-s1231']['primary_inference_depth'];futures=[]
            first=fit(arm,'first');gradient(arm,'stable1B',first,1920)
            for label,update in [('stable301M',576),('stable1B',1920)]:
                futures.append(cpu_pool.submit(score,arm,label,first,update,depth))
            end=fit(arm,'finish',first);gradient(arm,'matched',end,plan['cases'][arm+'-main-s1231']['total_updates'])
            futures.append(cpu_pool.submit(score,arm,'matched',end,end[1]['update'],depth))
            equal=fit(arm,'equal',first);gradient(arm,'equal',equal,1920)
            futures.append(cpu_pool.submit(score,arm,'equal',equal,1920,depth))
            if arm=='poisson2':
                for extra in [1,3]:
                    futures.append(cpu_pool.submit(score,arm,'matched',end,end[1]['update'],extra))
                    futures.append(cpu_pool.submit(score,arm,'equal',equal,1920,extra))
            for future in futures:future.result(timeout=max(1,deadline-time.time()))
        except BaseException:failed.set();raise
    result={'status':'STOPPED_FROZEN_LOW_DEPTH_MAIN_PILOT'}
    try:
        frozen()
        while managed_running():
            stopped();write_state(public/'status.json',{'status':'WAITING_FOR_PRECEDING_OWNERS','assigned_chips':0});time.sleep(5)
        write_state(public/'status.json',{'status':'RUNNING_FROZEN_LOW_DEPTH_MAIN_PILOT','maximum_chips':16,'started_unix':execution['started_unix']})
        with cf.ThreadPoolExecutor(3) as pool:
            for future in cf.as_completed([pool.submit(arm_worker,arm) for arm in ['ordinary','poisson2','fixed1']]):future.result()
        frozen();stopped()
        if len(native_records)!=9 or len(gradient_records)!=9 or len(scores)!=16:raise AssertionError('Main DAG did not finish every registered stage')
        by_key={(r['arm'],r['readout'],r['depth']):r['metrics']['weighted_nll'] for r in scores}
        contrasts={}
        for readout in ['matched','equal']:
            candidate=by_key['poisson2',readout,2]
            contrasts[readout]={'one_pass_minus_candidate':by_key['fixed1',readout,1]-candidate,
                'ordinary_minus_candidate':by_key['ordinary',readout,1]-candidate,
                'candidate_depth1_minus_depth2':by_key['poisson2',readout,1]-candidate,
                'candidate_depth1_minus_depth3':by_key['poisson2',readout,1]-by_key['poisson2',readout,3]}
        result={'status':'COMPLETE_FROZEN_LOW_DEPTH_MAIN_PILOT','contrasts':contrasts,
                'scope':'Execution recovery of unchanged one-seed consumed-DEV pilot;completed1B prefixes adopted exactly,ordinary restarts from its same fresh initialization after a failed70-update path with no checkpoint. Not terminal significance,confirmation or a strongest-baseline efficiency claim.'}
    except BaseException as error:failed.set();result.update(error=str(error));raise
    finally:
        failed.set()
        for identity,container in list(active.items()):
            try:cancel_worker_container(container,identity,'main_paired_stop')
            except BaseException:pass
        for process in list(cpu_active.values()):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        cpu_pool.shutdown(wait=True,cancel_futures=True)
        result.update(plan_sha256=plan_sha,wall_seconds=time.time()-execution['started_unix'],
            native_records=native_records,gradients=gradient_records,scores=scores,test_accessed=False,new_terminal_attempt=False)
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        write_state(public/'status.json',{'status':result['status'],'assigned_chips':0,'observed_unix':time.time()})
        write_state(public/'execution-costs.json',{'native_attempts':costs,'cpu_phases':cpu_costs,
            'wall_seconds':time.time()-execution['started_unix'],'scope':'All launched main fits,extra cooldown branches,zero-update full-gradient captures,CPU audits/scoring,cold compilation/checkpoint/recovery and failed paths. Conceptual model work and total research cost are distinct; concurrent durations are not summed reservation time.'})


if __name__=='__main__':main()
