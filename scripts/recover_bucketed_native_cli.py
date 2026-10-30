"""Repair a zero-update launcher-argument failure within the original cap."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_bucketed_native_qualification as original

ID='bucketed-native-qualification-cli-recovery-v1'
PUBLIC=Path('results/gradient_bucketing_20261030/native-cli-recovery-v1')


def prepare(root):
    from compute_trainium.execution_isolation import managed_running
    from compute_trainium.native_comm_ports import reserve_ports
    from compute_trainium.native_inode_projection import project_snapshot_inodes
    from compute_trainium.native_snapshot_manifest import split_snapshot_sources
    if managed_running():raise RuntimeError('Exclusive ownership required')
    old=root/original.ID
    read=lambda f:json.loads(f.read_text())
    p=copy.deepcopy(read(old/'plan.json'));result=read(old/'report.json')
    if result['status']!='STOPPED_BUCKETED_NATIVE_QUALIFICATION' or result['records']:
        raise AssertionError('Only the zero-update original launch may be recovered')
    log=(root/'logs'/(original.ID+'-warm.log')).read_text()
    if 'unrecognized arguments: --corpus /work/data-v1/corpus.json' not in log:
        raise AssertionError('Exact initiating parser failure required')
    if list((root/'runs'/(original.ID+'-warm')).glob('PRIVATE-*.pt')):
        raise AssertionError('Do not recover an unreviewed partial update history')
    changed=[f for f,h in p['source_sha256'].items() if file_sha256(f)!=h]
    if changed!=['scripts/qualify_bucketed_native.py']:
        raise AssertionError('Only the worker CLI may change')
    now=time.time();deadline=p['absolute_deadline_unix']
    if now+1800>deadline:raise TimeoutError('One bounded repair must fit original deadline')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<39*1024**3:
        raise RuntimeError('Full artifact allowance and reserve required')
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Snapshot reserve')
    sources={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in p['source_sha256']:
        if f not in sources:sources[f]=file_sha256(f)
    for f in ('docs/BUCKETED_NATIVE_CLI_RECOVERY_20261030.md','tests/test_bucketed_launcher_cli.py'):
        sources[f]=file_sha256(f)
    worker,host=split_snapshot_sources(sources);names=[ID+'-'+v for v in ('warm','replay')]
    p.update(identity=ID,registered_unix=now,seconds_limit=math.floor(deadline-now),
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},
        warm_phase=names[0],inode_projection=projection,
        recovery=dict(original_plan_sha256=file_sha256(old/'plan.json'),
            original_result_sha256=file_sha256(old/'report.json'),
            original_costs_sha256=file_sha256(original.PUBLIC/'execution-costs.json'),
            scope=Path('docs/BUCKETED_NATIVE_CLI_RECOVERY_20261030.md').read_text()))
    out=root/ID
    if out.exists() or PUBLIC.exists():raise FileExistsError('Retain all attempts')
    out.mkdir();PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    write_state(PUBLIC/'status.json',dict(status='REGISTERED_CLI_RECOVERY_NOT_STARTED',
        original_deadline_retained=True,scientific_fit_admitted=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True)
    a.add_argument('--action',choices=('prepare','run'),required=True);v=a.parse_args()
    if v.action=='prepare':prepare(v.root)
    else:
        original.ID=ID;original.PUBLIC=PUBLIC;original.run(v.root)
