"""Exclusive all-chip numerical fixture under an original, nonrenewing cap."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from compute_trainium.execution_isolation import cancel_worker_container, managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.owned_native_all_chips import create_all_chip_worker
from compute_trainium.owned_native_compact import reclaim_host
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='scale-sharding-fixture-v1'
PUBLIC=Path('results/scale_sharding_fixture_20261027')
SCALE_START=1793066985.0  # October 27, 02:09:45 UTC; includes prior preparation.
SCALE_END=SCALE_START+24*3600


class AllChipExecution(Execution):
    def check(self):
        super().check()
        require_inode_reserve([self.root,Path('/')])
        if time.time()>SCALE_END: raise TimeoutError('Original 24-hour scale feasibility cap')

    def native(self,identity,module,options,status):
        container=process=None;primary=None
        try:
            self.check()
            if (self.root/'runs'/identity).exists():
                raise FileExistsError('No automatic partial fixture retry')
            reclaim_host();container=create_all_chip_worker(self.root,identity)
            self.native_active[identity]=container
            limit=min(self.deadline,time.time()+self.native_seconds_limit(identity),SCALE_END)
            env=dict(os.environ,RESEARCH_ROOT=str(self.root),RESEARCH_CONTAINER_ROOT='/work/instance-nvme',
                RESEARCH_RUN=identity,RESEARCH_OWNER=identity,RESEARCH_CONTAINER=container,RESEARCH_ENTRY_MODULE=module,
                RESEARCH_DDP_RANKS='64',NEURON_RT_VISIBLE_CORES=','.join(map(str,range(64))),
                RESEARCH_TIMEOUT_SECONDS=str(max(1,int(limit-time.time()))))
            with (self.out/(identity+'.log')).open('x') as log:
                process=subprocess.Popen([os.sys.executable,'-u','-m','scripts.supervise_isolated',*options],
                    env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                self.check()
                if time.time()>limit: raise TimeoutError('All-chip fixture phase cap')
                time.sleep(2)
            if process.returncode: raise RuntimeError('All-chip fixture worker failed: '+identity)
            result=json.loads((self.root/'runs'/identity/'report.json').read_text())
            if result['status']!=status or result['world']!=64:raise AssertionError('Wrong fixture result')
            return result
        except BaseException as error:
            primary=error;self.failed.set();raise
        finally:
            if container:
                try:
                    cancel_worker_container(container,identity,'finite_scale_fixture_complete')
                    self.native_active.pop(identity,None)
                except BaseException as error:
                    self.cleanup_errors.append(dict(identity=identity,error=str(error),primary_error=str(primary)))
                    if primary is None:raise
            if process and process.poll() is None:
                process.terminate();process.wait(timeout=15)


def prepare(root):
    from compute_trainium.scale_sharded_candidate import scale_inventory
    if managed_running():raise RuntimeError('No fixture registration over an active owner')
    projection=project_snapshot_inodes(Path('.'),2)
    for filesystem in [root,Path('/'),root.parent]:
        if shutil.disk_usage(filesystem).free<36*1024**3:
            raise RuntimeError('Protected 35GiB reserve plus complete 1GiB fixture allowance required')
    if os.statvfs(root).f_favail<projection['required_free_inodes']:
        raise RuntimeError('Both complete fixture snapshots and artifacts must fit inode reserve')
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))
    if available<(512+128)*1024**3:
        raise RuntimeError('Protected host reserve plus exclusive worker cap required')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve existing registrations')
    if time.time()+1800>SCALE_END:raise TimeoutError('Complete fixture allowance does not fit scale cap')
    gate=Path('results/head_free_fixed2_numerical_v2_20261027/recovery/report.json')
    if json.loads(gate.read_text())['status']!='PASS_HEAD_FREE_FIXED_TWO_NATIVE_QUALIFICATION':
        raise AssertionError('Prior complete numerical closure required')
    sources={}
    for directory in ['compute_trainium','scripts','configs']:
        for file in Path(directory).rglob('*.py'):
            if file.is_file():sources[str(file)]=file_sha256(file)
    for file in ['docs/SCALE_FEASIBILITY_20261027.md','tests/test_scale_sharded_candidate.py',
                 'tests/test_owned_native_all_chips.py','scripts/run_native_isolated.sh',
                 'third_party/torchtitan_neuron_sharding/LICENSE']:
        sources[file]=file_sha256(file)
    worker,host=split_snapshot_sources(sources)
    names=[IDENTITY+'-warm',IDENTITY+'-replay']
    plan=dict(identity=IDENTITY,purpose='scale_sharded_software_fixture_only',registered_unix=time.time(),
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        world=64,seconds_limit=1800,phase_seconds_limit=900,cpu_phase_seconds_limit=60,
        artifact_allowance_bytes=1024**3,inode_projection=projection,
        root_comm_ports=reserve_ports(root,names),phases={n:dict(mode=mode) for n,mode in zip(names,['warm','replay'])},
        scale_budget_started_unix=SCALE_START,scale_budget_deadline_unix=SCALE_END,
        research_examples=0,research_optimizer_updates=0,actual_fixture_optimizer_updates=4,
        scientific_fit_admitted=False,terminal_alpha_spent=0,model_inventory=scale_inventory(),
        prior_numerical_report_sha256=file_sha256(gate),scope=Path('docs/SCALE_FEASIBILITY_20261027.md').read_text())
    out.mkdir();PUBLIC.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    write_state(PUBLIC/'status.json',dict(status='PREPARED_NOT_ACTIVATED',scientific_fit_admitted=False))


def run(root):
    out=root/IDENTITY;plan_file=out/'plan.json';execution=AllChipExecution(root,plan_file,out/'execution')
    result=dict(status='STOPPED_SCALE_SHARDED_SOFTWARE_FIXTURE',records={},scientific_fit_admitted=False)
    def stop(number,frame):raise KeyboardInterrupt('Scale fixture interrupted')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        for mode in ['warm','replay']:
            name=IDENTITY+'-'+mode
            options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
                '--plan-sha',execution.sha,'--mode',mode,'--comm-port',str(execution.plan['root_comm_ports'][name])]
            if mode=='replay':options+=['--parent','/work/instance-nvme/runs/'+IDENTITY+'-warm']
            result['records'][mode]=execution.native(name,'scripts.qualify_scale_sharding',options,'PASS_SCALE_SHARDED_SOFTWARE_FIXTURE')
            write_state(PUBLIC/'progress.json',result)
        result['status']='PASS_SCALE_SHARDED_SOFTWARE_FIXTURE_AND_REPLAY'
    except BaseException as error:
        result.update(error=str(error),error_type=type(error).__name__)
        raise
    finally:
        result.update(plan_sha256=execution.sha,research_examples=0,research_optimizer_updates=0,
            scope='All-chip small-array qualification only; no large scientific fit or scale result.')
        execution.close(out/'execution-costs.json')
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        write_state(PUBLIC/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))
        write_state(PUBLIC/'status.json',dict(status=result['status'],scientific_fit_admitted=False))
        for s,handler in handlers.items():signal.signal(s,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    run(args.root) if args.run else prepare(args.root)
