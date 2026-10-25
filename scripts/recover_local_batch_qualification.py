"""Recover a pre-update mount-path failure under the original finite deadline."""
import argparse
import copy
import json
import os
from pathlib import Path

from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_local_batch_qualification as original
from scripts import run_mixing_control_qualification as engine

IDENTITY='local-batch-qualification-v2'
PUBLIC=Path('results/local_batch_qualification_20261025/recovery-v2')


def portable_asset_links(previous,preparation):
    for file in previous.iterdir():
        if file.is_file() and file.name!='manifest.json':
            target=os.path.relpath(file.resolve(),preparation.resolve())
            (preparation/file.name).symlink_to(target)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root;old_out=root/'local-batch-qualification-v1'
    old=json.loads((old_out/'plan.json').read_text());failure=json.loads((old_out/'report.json').read_text())
    clock=json.loads((old_out/'execution/execution-start.json').read_text())
    if failure['status']!='STOPPED_CHEAP_MIXING_CONTROL_QUALIFICATION' or failure['records']:
        raise ValueError('Only preserved failed pre-update capture eligible')
    for key in old['cases']:
        run=root/'runs'/('local-batch-qualification-v1-'+key+'-warm')
        if any(run.glob('*.pt')) or (run/'report.json').exists():raise ValueError('Unexpected fitted state;no silent restart')
    out=root/IDENTITY;out.mkdir();preparation=out/'preparation';preparation.mkdir()
    previous=root/json.loads((root/'future-span-math-qualification-v2/plan.json').read_text())['preparation']
    portable_asset_links(previous,preparation)
    manifest=json.loads((root/old['preparation']/'manifest.json').read_text())
    write_state(preparation/'manifest.json',manifest)
    names=[IDENTITY+'-'+key+'-'+mode for key in old['cases'] for mode in ['warm','replay']]
    sources={**old['source_sha256'],**{p:file_sha256(p) for p in ['scripts/recover_local_batch_qualification.py','tests/test_portable_asset_links.py']}}
    worker,host=split_snapshot_sources(sources)
    plan=copy.deepcopy(old);plan.update(identity=IDENTITY,source_sha256=sources,worker_source_sha256=worker,
        host_only_source_sha256=host,preparation=str(preparation.relative_to(root)),
        preparation_manifest_sha256=file_sha256(preparation/'manifest.json'),root_comm_ports=reserve_ports(root,names),
        original_deadline_unix=clock['deadline_unix'],failed_plan_sha256=file_sha256(old_out/'plan.json'),
        failed_report_sha256=file_sha256(old_out/'report.json'),
        recovery_scope='Only container-portable relative asset links and new owned identities. First attempt failed before model initialization/updates. Original clock,binary assets,math,local batches and numerical thresholds preserved. Failed assigned costs remain separate;reservation opportunity overlaps the original admission and must not be added twice.')
    PUBLIC.mkdir(parents=True,exist_ok=False);file=out/'plan.json';write_state(file,plan);write_state(PUBLIC/'plan.json',plan)
    execution=out/'execution';execution.mkdir();write_state(execution/'execution-start.json',{**clock,'plan_sha256':file_sha256(file)})
    engine.IDENTITY=IDENTITY;engine.PUBLIC=PUBLIC;engine.Execution=original.GeometryExecution
    engine.run(root,plan,resume=True)
    path=out/'report.json';result=json.loads(path.read_text())
    if result['status']!='PASS_COMPLETE_CHEAP_MIXING_CONTROL_QUALIFICATION':raise ValueError('Bounded numerical workflow failed')
    result.update(status='PASS_LOCAL_BATCH_GEOMETRY_QUALIFICATION',scientific_training_admitted=False)
    write_state(path,result);write_state(PUBLIC/'report.json',result);write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


if __name__=='__main__':main()
