"""Bounded all-chip proof of the new checkpoint-lineage wrapper only."""
import argparse
import json
import signal
from pathlib import Path
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.execution_isolation import managed_running
from scripts.run_scale_sharding_fixture import AllChipExecution,SCALE_END

IDENTITY='scale-training-state-gate-v1'
PUBLIC=Path('results/scale_training_state_gate_20261027')


def prepare(root):
    if managed_running():raise RuntimeError('Prior Native owners must be terminal')
    for native,status in [('scale-376m-fixed-two-numerical-v1','PASS_SCALE_MEDIUM_FIXED_TWO_COMPLETE_NUMERICAL_QUALIFICATION'),
                          ('scale-375m-ordinary-numerical-v1','PASS_SCALE_MEDIUM_ORDINARY_COMPLETE_NUMERICAL_QUALIFICATION')]:
        if json.loads((root/native/'completion-verification.json').read_text())['status']!=status:
            raise ValueError('Both actual-shape complete numerical proofs required')
    folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve state-gate attempts')
    sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs','third_party'] for f in Path(d).rglob('*.py') if f.is_file()}
    for f in ['tests/test_scale_training_fixture.py','tests/test_scale_training_state.py','tests/test_scale_work_ledger.py',
              'scripts/run_native_isolated.sh','docs/SCALE_JOINT_DEVELOPMENT_20261027.md']:sources[f]=file_sha256(f)
    worker,host=split_snapshot_sources(sources);names=[IDENTITY+'-'+m for m in ['warm','replay']];start=time.time()
    plan=dict(identity=IDENTITY,purpose='scale_training_state_software_fixture_only',world=64,worker_memory_gib=128,
        seconds_limit=1200,phase_seconds_limit=600,cpu_phase_seconds_limit=60,absolute_deadline_unix=min(start+1200,SCALE_END),
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        phases={n:dict(mode=m) for n,m in zip(names,['warm','replay'])},root_comm_ports=reserve_ports(root,names),
        research_examples=0,research_optimizer_updates=0,actual_fixture_optimizer_updates=3,
        scientific_fit_admitted=False,registered_unix=start,scale_budget_deadline_unix=SCALE_END,
        scope='Only new lineage/checkpoint wrapper on small software arrays. Both actual medium model/loss/global512/optimizer/precision paths are separately qualified. Four-process CPU wrapper save/restore/replay is exact. This gate changes no mathematical kernel and admits no research examples.')
    folder.mkdir();PUBLIC.mkdir();write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
    folder=root/IDENTITY;execution=AllChipExecution(root,folder/'plan.json',folder/'execution');records={}
    result=dict(status='STOPPED_SCALE_TRAINING_STATE_GATE',scientific_fit_admitted=False)
    def stop(number,frame):raise KeyboardInterrupt('State gate interrupted')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        for mode in ['warm','replay']:
            name=IDENTITY+'-'+mode
            options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
                '--plan-sha',execution.sha,'--mode',mode,'--comm-port',str(execution.plan['root_comm_ports'][name])]
            if mode=='replay':options+=['--parent','/work/instance-nvme/runs/'+IDENTITY+'-warm']
            records[mode]=execution.native(name,'scripts.qualify_scale_training_state',options,'PASS_SCALE_TRAINING_STATE_SOFTWARE_FIXTURE')
        replay=json.loads((root/'runs'/namesafe('replay')/'result.json').read_text())
        if replay['weights']['maximum_absolute_error'] or replay['moments']['maximum_absolute_error']:
            raise AssertionError('Exact fresh-process lineage-wrapper replay required')
        result.update(status='PASS_SCALE_REGISTERED_TRAINER_AND_REPLAY',replay=replay,
            scope='New state wrapper proof only; existing actual-shape full512/first-update/moments/warm proofs remain prerequisites. No new large scientific fit admitted.')
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(records=records,plan_sha256=execution.sha,actual_fixture_optimizer_updates=sum(r['actual_optimizer_updates'] for r in records.values()),research_optimizer_updates=0)
        execution.close(folder/'execution-costs.json');write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',result)
        for s,handler in handlers.items():signal.signal(s,handler)


def namesafe(mode):return IDENTITY+'-'+mode


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
