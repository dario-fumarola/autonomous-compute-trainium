"""Finite whole-training performance check before any optimizer adoption."""
import argparse
import concurrent.futures as cf
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

IDENTITY='owner-full-training-screen-v1'
PUBLIC=Path('results/owner_full_training_screen_20261025')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();out=args.root/IDENTITY;file=out/'plan.json'
    if not file.exists():
        parent_file=args.root/'future-span-math-qualification-v2/plan.json'
        old=json.loads(parent_file.read_text());proof=args.root/'owner-muon-fixture-native-v1/report.json'
        if json.loads(proof.read_text())['status']!='COMPLETE_OWNER_MUON_FIXTURE_SCREEN' or managed_running():
            raise RuntimeError('Complete matrix screen and idle ownership required')
        sources={**old['source_sha256'],**{p:file_sha256(p) for p in [
            'scripts/qualify_owner_training_screen_native.py','scripts/run_owner_training_screen.py',
            'compute_trainium/owner_broadcast_muon_candidate.py','tests/test_owner_broadcast_muon_candidate.py']}}
        worker,host=split_snapshot_sources(sources);variants={f'one-ntp-{v}':v for v in ['replicated','owner']}
        plan=copy.deepcopy(old);plan.update(identity=IDENTITY,registered_unix=time.time(),
            cases={k:old['cases']['one-ntp'] for k in variants},optimizer_variants=variants,
            source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
            root_comm_ports=reserve_ports(args.root,[IDENTITY+'-'+k for k in variants]),
            matrix_screen_sha256=file_sha256(proof),seconds_limit=900,phase_seconds_limit=600,cpu_phase_seconds_limit=600,
            scope='Two simultaneous eight-chip one-pass full-training throughput screens using existing qualification-only initialization,natural TRAIN windows and100 warm updates. No scientific weights,held-out data,teacher or new objective. Physical execution only;no adoption. Less than5% observed whole-training improvement closes this performance proposal at this stage. A larger gain still requires swapped-hardware timing,CPU numerical and complete recovery gates before adoption.')
        out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False);write_state(file,plan);write_state(PUBLIC/'plan.json',plan)
    plan=json.loads(file.read_text());execution=Execution(args.root,file,out/'execution')
    result=dict(status='STOPPED_OWNER_FULL_TRAINING_SCREEN')
    def job(key):
        name=IDENTITY+'-'+key
        return execution.native(name,'scripts.qualify_owner_training_screen_native',
            ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
             '--plan-sha',execution.sha,'--case',key,'--mode','warm','--comm-port',str(plan['root_comm_ports'][name])],
            'CAPTURED_FUTURE_SPAN_TRAINING_QUALIFICATION')
    try:
        with cf.ThreadPoolExecutor(2) as pool:
            tasks={k:pool.submit(job,k) for k in plan['cases']}
            records={plan['optimizer_variants'][k]:f.result() for k,f in tasks.items()}
        if any(not r['all_complete_replica_states_equal'] or r['warm_updates']!=100 for r in records.values()):
            raise AssertionError('Full training screen proof/timing incomplete')
        ratio=records['owner']['warm_seconds']/records['replicated']['warm_seconds']
        result=dict(status='COMPLETE_OWNER_FULL_TRAINING_SCREEN',records=records,
            owner_to_replicated_warm_time=ratio,clears_five_percent_screen=ratio<=.95,
            native_final_model_optimizer_digests_equal=records['owner']['final_model_optimizer_digest']==records['replicated']['final_model_optimizer_digest'],
            independent_cpu_numerics_complete=False,full_recovery_qualified=False,
            hardware_swapped_replication_complete=False,scientific_adoption=False)
    except BaseException as error:result['error']=str(error);raise
    finally:
        execution.close(PUBLIC/'execution-costs.json')
        result.update(plan_sha256=execution.sha,qualification_only=True,research_optimizer_updates=0,
            test_accessed=False,terminal_alpha_spent=0,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':main()
