"""Owned bounded execution shared by future scientific/driver queues."""
import fcntl
import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

from .execution_isolation import managed_running,cancel_worker_container
from .owned_native_compact import create_compact_worker,reclaim_host
from .queue import write_state
from .training import file_sha256


class Execution:
    def __init__(self,root,plan_file,out,*,resume=False):
        self.root,self.plan_file,self.out=Path(root),Path(plan_file),Path(out)
        self.plan=json.loads(self.plan_file.read_text());self.sha=file_sha256(self.plan_file)
        self.out.mkdir(parents=True,exist_ok=True)
        self.lock_file=(self.out/'controller.lock').open('a')
        fcntl.flock(self.lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.failed=threading.Event();self.lock=threading.Lock();self.native_active={};self.cpu_active={};self.cpu_attempts=[]
        self.halves=queue.Queue()
        for half in [0,1]:self.halves.put(half)
        record=self.out/'execution-start.json'
        if record.exists():
            if not resume:raise FileExistsError('Existing execution requires explicit resume')
            self.execution=json.loads(record.read_text())
            if self.execution['plan_sha256']!=self.sha:raise ValueError('Resume changed the exact saved plan')
        else:
            if resume:raise ValueError('No prior execution exists to resume')
            if managed_running():raise RuntimeError('Campaign admission requires predecessor completion and no active native owners')
            start=time.time();self.execution={'started_unix':start,'deadline_unix':start+self.plan['seconds_limit'],'plan_sha256':self.sha}
            write_state(record,self.execution)
        self.deadline=self.execution['deadline_unix'];self.check()

    def check(self):
        if self.failed.is_set() or time.time()>self.deadline:raise TimeoutError('Original execution deadline or paired stop')
        if file_sha256(self.plan_file)!=self.sha or any(file_sha256(p)!=sha for p,sha in self.plan['source_sha256'].items()):
            raise AssertionError('Frozen exact source list/plan changed')

    def native(self,identity,module,options,status):
        half=container=process=None
        try:
            self.check();directory=self.root/'runs'/identity
            if (directory/'report.json').exists():
                result=json.loads((directory/'report.json').read_text())
                closure=self.root/'snapshots'/identity/'closure.json'
                if not closure.exists() or json.loads(closure.read_text())['exit_code']!=0:raise ValueError('Reused native phase lacks successful closure')
            else:
                if directory.exists():raise RuntimeError('Partial native phase needs explicit recovery registration: '+identity)
                while half is None:
                    self.check()
                    try:half=self.halves.get(timeout=5)
                    except queue.Empty:pass
                reclaim_host();container=create_compact_worker(self.root,identity,half)
                with self.lock:self.native_active[identity]=container
                limit=min(self.deadline,time.time()+self.plan['phase_seconds_limit'])
                env=dict(os.environ,RESEARCH_ROOT=str(self.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                    RESEARCH_RUN=identity,RESEARCH_OWNER=identity,RESEARCH_CONTAINER=container,RESEARCH_ENTRY_MODULE=module,
                    RESEARCH_DDP_RANKS='32',NEURON_RT_VISIBLE_CORES=','.join(map(str,range(half*32,half*32+32))),
                    RESEARCH_TIMEOUT_SECONDS=str(max(1,int(limit-time.time()))))
                with (self.out/(identity+f'-{int(time.time())}.log')).open('x') as log:
                    process=subprocess.Popen([os.sys.executable,'-u','-m','scripts.supervise_isolated',*options],
                        env=env,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    self.check()
                    if time.time()>limit:raise TimeoutError('Native phase cap: '+identity)
                    time.sleep(5)
                if process.returncode:raise RuntimeError('Native phase failed: '+identity)
                result=json.loads((directory/'report.json').read_text())
            if result['status']!=status:raise ValueError('Native result has unexpected status: '+identity)
            return result
        except BaseException:self.failed.set();raise
        finally:
            if container:
                cancel_worker_container(container,identity,'finite_future_phase_complete')
                with self.lock:self.native_active.pop(identity,None)
            if process and process.poll() is None:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.terminate()
            if half is not None:self.halves.put(half)

    def cpu(self,identity,module,options,report_file,status):
        process=None;started=time.time();report_file=Path(report_file);reused=report_file.exists()
        try:
            self.check()
            if reused:result=json.loads(report_file.read_text())
            else:
                if report_file.parent.exists():raise RuntimeError('Partial CPU phase needs explicit recovery: '+identity)
                report_file.parent.parent.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,USE_TORCH_XLA='0',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12')
                with (self.out/(identity+f'-{int(time.time())}.log')).open('x') as log:
                    process=subprocess.Popen([os.sys.executable,'-u','-m',module,*options],env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with self.lock:self.cpu_active[identity]=process
                limit=min(self.deadline,started+self.plan['cpu_phase_seconds_limit'])
                while process.poll() is None:
                    self.check()
                    if time.time()>limit:raise TimeoutError('CPU phase cap: '+identity)
                    time.sleep(5)
                if process.returncode:raise RuntimeError('CPU phase failed: '+identity)
                result=json.loads(report_file.read_text())
            if result['status']!=status:raise ValueError('CPU result has unexpected status: '+identity)
            return result
        except BaseException:self.failed.set();raise
        finally:
            if process and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            with self.lock:
                self.cpu_active.pop(identity,None)
                self.cpu_attempts.append({'identity':identity,'assigned_chips':0,'wall_seconds':time.time()-started,
                    'reused':reused,'exit_code':process.returncode if process else None})

    def close(self,cost_file):
        self.failed.set()
        for identity,container in list(self.native_active.items()):
            try:cancel_worker_container(container,identity,'finite_future_execution_stop')
            except BaseException:pass
        for process in list(self.cpu_active.values()):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        attempts=[]
        for identity in self.plan['root_comm_ports']:
            folder=self.root/'snapshots'/identity;launch=folder/'launch.json';closure=folder/'closure.json'
            if not launch.exists():continue
            if closure.exists():attempts.append({'identity':identity,**json.loads(closure.read_text())})
            else:
                row=json.loads(launch.read_text());wall=max(0,time.time()-row['started_unix'])
                attempts.append({'identity':identity,'closure_missing':True,'wall_seconds_through_collection':wall,
                    'assigned_chip_hours_through_collection':wall*len(row['physical_chips'])/3600})
        write_state(cost_file,{'native_attempts':attempts,'current_controller_cpu_attempts':self.cpu_attempts,
            'execution':self.execution,'wall_seconds_since_admission':time.time()-self.execution['started_unix'],
            'whole_instance_opportunity_chip_hours':(time.time()-self.execution['started_unix'])*16/3600,
            'scope':'Each registered native phase closure once,including failures and completed work reused after controller resume. CPU reports retain their intrinsic durations;current-controller attempts distinguish reuse. Reservation opportunity is separate from assigned chips and must not be double-added across overlapping phases.'})
        self.lock_file.close()
