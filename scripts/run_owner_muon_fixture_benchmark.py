"""Bounded performance screen using idle chips;no scientific adoption."""
import argparse
import concurrent.futures as cf
import copy
import json
import time
from pathlib import Path

import torch
from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_candidate import parameter_groups
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='owner-muon-fixture-native-v1'
PUBLIC=Path('results/owner_muon_fixture_native_20261025')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    out=args.root/IDENTITY;file=out/'plan.json'
    if not file.exists():
        parent_file=args.root/'small-core-mixing-main-v2/plan.json';parent=json.loads(parent_file.read_text())
        # All registered scientific native phases must have ended;CPU readouts
        # may continue. Neither controller can cancel the other's containers.
        for name in parent['root_comm_ports']:
            closure=args.root/'snapshots'/name/'closure.json'
            if not closure.exists() or json.loads(closure.read_text())['exit_code']!=0:
                raise RuntimeError('Every original scientific native phase must close first')
        if managed_running():raise RuntimeError('No active native worker may be displaced')
        with torch.device('meta'):
            model=FutureSpanCandidate(ParcaeCandidateConfig(**parent['model']),
                FutureSpanConfig(**parent['future_config']),head_seed=1,initialize=False,fp16_body=False)
        groups=parameter_groups(model.named_parameters(),gate_policy='release_muon')[1:]
        geometry=[dict(shape=list(g['params'][0].shape),count=len(g['params'])) for g in groups]
        additions=['scripts/benchmark_owner_muon_native.py','scripts/run_owner_muon_fixture_benchmark.py',
            'compute_trainium/owner_broadcast_muon_candidate.py','tests/test_owner_broadcast_muon_candidate.py']
        sources={**parent['source_sha256'],**{p:file_sha256(p) for p in additions}}
        worker,host=split_snapshot_sources(sources)
        names=[IDENTITY+'-'+v for v in ['replicated','owner']]
        plan=dict(identity=IDENTITY,registered_unix=time.time(),source_sha256=sources,
            worker_source_sha256=worker,host_only_source_sha256=host,qualification_only=True,
            matrix_groups=geometry,optimizer_updates=120,research_examples=0,
            original_main_plan_sha256=file_sha256(parent_file),seconds_limit=1200,
            phase_seconds_limit=600,cpu_phase_seconds_limit=600,root_comm_ports=reserve_ports(args.root,names),
            scope='Matrix-only software-fixture performance screen during the completed scientific native queue CPU tail. Same actual Muon group shapes,FP32 recipe,changing schedules,100 timed warm updates. Neither full model training nor AdamW measured. Cross-variant weights/moments and complete native replica equality required. No scientific state adoption,new source data,held-out access,TEST or terminal alpha.')
        out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False)
        write_state(file,plan);write_state(PUBLIC/'plan.json',plan)
    plan=json.loads(file.read_text())
    if args.prepare_only:return
    execution=Execution(args.root,file,out/'execution');result=dict(status='STOPPED_OWNER_MUON_FIXTURE_SCREEN')
    def job(variant):
        name=IDENTITY+'-'+variant
        return execution.native(name,'scripts.benchmark_owner_muon_native',
            ['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
             '--plan-sha',execution.sha,'--variant',variant,'--comm-port',str(plan['root_comm_ports'][name])],
            'COMPLETE_OWNER_MUON_FIXTURE_BENCHMARK')
    try:
        with cf.ThreadPoolExecutor(2) as pool:
            tasks={v:pool.submit(job,v) for v in ['replicated','owner']}
            records={v:f.result() for v,f in tasks.items()}
        errors=[]
        def compare(a,b):
            if isinstance(a,torch.Tensor):
                x,y=a.double(),b.double();den=float(torch.linalg.vector_norm(x))
                rel=float(torch.linalg.vector_norm(x-y))/max(den,1e-30)
                errors.append(rel)
                if rel>2e-5 or not torch.isfinite(y).all():raise AssertionError('Cross-variant optimizer state differs')
            elif isinstance(a,dict):
                if a.keys()!=b.keys():raise AssertionError('State schema changed')
                for k in a:compare(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                if len(a)!=len(b):raise AssertionError('State length changed')
                for x,y in zip(a,b,strict=True):compare(x,y)
            elif a!=b:raise AssertionError('State metadata changed')
        for step in [1,120]:
            files=[args.root/'runs'/(IDENTITY+'-'+v)/f'state-u{step}.pt' for v in ['replicated','owner']]
            compare(*(torch.load(p,map_location='cpu',mmap=True,weights_only=False) for p in files))
        result=dict(status='COMPLETE_OWNER_MUON_FIXTURE_SCREEN',records=records,
            max_cross_variant_tensor_relative_l2=max(errors),
            owner_to_replicated_warm_time=records['owner']['warm_seconds']/records['replicated']['warm_seconds'],
            full_training_qualified=False,scientific_adoption=False)
    except BaseException as error:result['error']=str(error);raise
    finally:
        execution.close(PUBLIC/'execution-costs.json')
        result.update(plan_sha256=execution.sha,test_accessed=False,research_examples=0,
            research_optimizer_updates=0,terminal_alpha_spent=0,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':main()
