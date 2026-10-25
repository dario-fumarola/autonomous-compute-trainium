"""Snapshot-policy recovery only;retain the original absolute TUNE deadline."""
import argparse
import copy
import json
import time
from pathlib import Path

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_mixing_control_tune as engine

IDENTITY='small-core-mixing-tune-v2'
PUBLIC=Path('results/small_core_mixing_tune_20261025/recovery-v2')


def register(root):
    old_file=root/'small-core-mixing-tune-v1/plan.json'
    old=json.loads(old_file.read_text())
    stopped_file=old_file.parent/'report.json';stopped=json.loads(stopped_file.read_text())
    start_file=old_file.parent/'execution/execution-start.json';start=json.loads(start_file.read_text())
    if (stopped['status']!='STOPPED_CHEAP_MIXING_TUNE' or stopped['training_records']
            or stopped['scores'] or stopped['gradient_gates'] or managed_running()):
        raise ValueError('Only a stopped-before-initialization run is eligible')
    for key in list(old['cases'])[:2]:
        directory=root/'runs'/('small-core-mixing-tune-v1-'+key+'-preflight')
        log=root/'logs'/('small-core-mixing-tune-v1-'+key+'-preflight.log')
        if directory.exists() or 'Exact complete saved executable source list required' not in log.read_text():
            raise AssertionError('Expected fail-closed pre-initialization failure only')
    if any(file_sha256(p)!=h for p,h in old['source_sha256'].items()):
        raise AssertionError('Preserve every original frozen file')
    remaining=start['deadline_unix']-time.time()
    if remaining<old['measured_budget']['conservative_seconds']:
        raise RuntimeError('Remaining original budget cannot fit unchanged experiment')
    additions=['scripts/train_mixing_control_scientific_v2.py','scripts/run_mixing_control_tune_v2.py',
               'tests/test_mixing_snapshot_v2.py']
    source={**old['source_sha256'],**{p:file_sha256(p) for p in additions}}
    worker,host=split_snapshot_sources(source)
    plan=copy.deepcopy(old)
    for case in plan['cases'].values():
        case['work_table']='../small-core-mixing-tune-v1/'+case['work_table']
    names=[IDENTITY+'-'+key+'-'+phase for key in plan['cases']
           for phase in ['preflight','preflight-gradient','finish','final-gradient']]
    plan.update(identity=IDENTITY,registered_unix=time.time(),source_sha256=source,
        worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),seconds_limit=remaining,
        original_absolute_deadline_unix=start['deadline_unix'],
        stopped_plan_sha256=file_sha256(old_file),stopped_report_sha256=file_sha256(stopped_file),
        recovery_scope='Launcher snapshot-policy correction only. CPU-only third_party stays frozen but is not required in the native copied tree. Same model,data,initialization,three rates,endpoints,decision rule and original absolute deadline. Original attempt retained and charged;no checkpoint adopted.')
    out=root/IDENTITY;out.mkdir(exist_ok=False);PUBLIC.mkdir(parents=True,exist_ok=False)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan


class RecoveryExecution(Execution):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.deadline=min(self.deadline,self.plan['original_absolute_deadline_unix'])
        self.execution['deadline_unix']=self.deadline
        write_state(self.out/'execution-start.json',self.execution)
        self.check()

    def native(self,identity,module,options,status):
        if module=='scripts.train_mixing_control_scientific':
            module='scripts.train_mixing_control_scientific_v2'
        return super().native(identity,module,options,status)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();file=args.root/IDENTITY/'plan.json'
    plan=json.loads(file.read_text()) if file.exists() else register(args.root)
    if args.prepare_only:return
    engine.IDENTITY=IDENTITY;engine.PUBLIC=PUBLIC;engine.Execution=RecoveryExecution
    engine.run(args.root,plan,resume=args.resume)


if __name__=='__main__':main()
