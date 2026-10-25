"""Reuse bounded numerical/recovery execution with larger local batches."""
import argparse
import copy
import json
from pathlib import Path
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span_execution import Execution
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_mixing_control_qualification as engine

IDENTITY='local-batch-qualification-v1'
PUBLIC=Path('results/local_batch_qualification_20261025')


class GeometryExecution(Execution):
    def native(self,identity,module,options,status):
        if module=='scripts.qualify_mixing_control_native':
            module='scripts.qualify_local_batch_native'
            key=options[options.index('--case')+1]
            options=[*options,'--local-batch',str(self.plan['local_batches'][key])]
        return super().native(identity,module,options,status)

    def cpu(self,identity,module,options,report_file,status):
        if module=='scripts.audit_mixing_control_cpu':module='scripts.audit_local_batch_cpu'
        return super().cpu(identity,module,options,report_file,status)


def register(root):
    old_file=root/'future-span-math-qualification-v2/plan.json';old=json.loads(old_file.read_text())
    prerequisite=root/'owner-full-training-screen-v1/report.json'
    if json.loads(prerequisite.read_text())['status']!='COMPLETE_OWNER_FULL_TRAINING_SCREEN' or managed_running():
        raise RuntimeError('Completed predecessor and idle native ownership required')
    additions=['compute_trainium/microbatch_geometry.py','tests/test_microbatch_geometry.py',
        'scripts/qualify_local_batch_native.py','scripts/audit_local_batch_cpu.py',
        'scripts/run_local_batch_qualification.py','docs/LOCAL_BATCH_GEOMETRY_QUALIFICATION_20261025.md']
    sources={**old['source_sha256'],**{p:file_sha256(p) for p in additions}}
    if any(file_sha256(p)!=h for p,h in sources.items()):raise AssertionError('Qualified source changed')
    worker,host=split_snapshot_sources(sources)
    out=root/IDENTITY;out.mkdir();preparation=out/'preparation';preparation.mkdir()
    previous=root/old['preparation'];manifest=json.loads((previous/'manifest.json').read_text())
    cases={f'one-b{batch}':{**old['cases']['one-ntp'],'identity':f'one-b{batch}'} for batch in [2,4]}
    # Only metadata identities change. Every actual immutable asset is reused.
    for file in previous.iterdir():
        if file.is_file() and file.name!='manifest.json':(preparation/file.name).symlink_to(file.resolve())
    manifest['cases']=list(cases.values());manifest['parent_manifest_sha256']=file_sha256(previous/'manifest.json')
    write_state(preparation/'manifest.json',manifest)
    names=[IDENTITY+'-'+key+'-'+mode for key in cases for mode in ['warm','replay']]
    plan=copy.deepcopy(old);plan.update(identity=IDENTITY,registered_unix=time.time(),cases=cases,
        local_batches={f'one-b{batch}':batch for batch in [2,4]},
        preparation=str(preparation.relative_to(root)),preparation_manifest_sha256=file_sha256(preparation/'manifest.json'),
        model_variant='fixed1_local_batch_geometry_fullgrad',source_sha256=sources,
        worker_source_sha256=worker,host_only_source_sha256=host,seconds_limit=1800,
        phase_seconds_limit=600,cpu_phase_seconds_limit=900,root_comm_ports=reserve_ports(root,names),
        prerequisite_sha256={str(prerequisite.relative_to(root)):file_sha256(prerequisite)},
        scope=Path('docs/LOCAL_BATCH_GEOMETRY_QUALIFICATION_20261025.md').read_text())
    PUBLIC.mkdir(parents=True,exist_ok=False);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    file=args.root/IDENTITY/'plan.json';plan=json.loads(file.read_text()) if file.exists() else register(args.root)
    if args.prepare_only:return
    engine.IDENTITY=IDENTITY;engine.PUBLIC=PUBLIC;engine.Execution=GeometryExecution
    engine.run(args.root,plan)
    path=args.root/IDENTITY/'report.json';result=json.loads(path.read_text())
    if result['status']!='PASS_COMPLETE_CHEAP_MIXING_CONTROL_QUALIFICATION':raise ValueError('Inherited bounded workflow did not pass')
    result['status']='PASS_LOCAL_BATCH_GEOMETRY_QUALIFICATION';result['scientific_training_admitted']=False
    result['scope']=plan['scope']
    write_state(path,result);write_state(PUBLIC/'report.json',result);write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


if __name__=='__main__':main()
