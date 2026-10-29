"""One explicit cached-graph retry inside the original qualification deadline."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
import scripts.run_chunked_persistent_qualification as original

ID='chunked-persistent-qualification-recovery-v1'
PUBLIC=Path('results/chunked_persistent_workspace_20261029/numerical-recovery-v1')


def retained_deadline_plan(old,now):
    remaining=int(old['absolute_deadline_unix']-now)
    if remaining<600:raise TimeoutError('At least ten minutes must remain inside the original deadline')
    p=copy.deepcopy(old)
    p.update(identity=ID,registered_unix=now,seconds_limit=remaining,
        warm_phase=ID+'-warm',original_qualification_identity=old['identity'],
        original_absolute_deadline_unix=old['absolute_deadline_unix'],
        recovery_reason='First attempt passed all four cold checks and100 warm updates but compilation exhausted its phase cap during replica capture. Rerun the unchanged disposable fixture from the original initialization using existing compiled graphs. No deadline, numerical, throughput or reserve relaxation; no scientific initialization reuse.')
    return p


def prepare(root):
    if managed_running():raise RuntimeError('Failed owner must be closed')
    if shutil.disk_usage(root).free<39*1024**3 or shutil.disk_usage('/').free<35*1024**3:
        raise RuntimeError('Four GiB additional artifacts plus protected reserves')
    oldroot=root/'chunked-persistent-qualification-v1';old=json.loads((oldroot/'plan.json').read_text())
    result=json.loads((oldroot/'report.json').read_text())
    if result.get('error_type')!='TimeoutError' or result.get('error')!='All-chip fixture phase cap':
        raise AssertionError('Only the diagnosed phase timeout is eligible')
    for f,h in old['source_sha256'].items():
        if file_sha256(f)!=h:raise AssertionError('Original qualified sources must stay unchanged')
    run=root/'runs'/old['warm_phase'];checks=json.loads((run/'cold-local-checks.json').read_text())
    if set(checks)!={'0','1','2','3'}:raise AssertionError('All four completed cold checks required')
    for c in checks.values():
        if abs(c['loss_error'])>.003 or max(v['relative_l2'] for v in c.values() if isinstance(v,dict))>.02:
            raise AssertionError('No retry of a numerical failure')
    updates=json.loads((run/'updates.json').read_text());warm=[r for r in updates if 4<=r['update']<=103]
    times={str(k):sum(r['seconds'] for r in warm if r['extra_steps']==k)/sum(r['extra_steps']==k for r in warm) for k in range(4)}
    if len(warm)!=100 or max(times.values())>1 or times['2']/times['0']>3:
        raise AssertionError('Completed original throughput must already pass')
    p=retained_deadline_plan(old,time.time())
    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/CHUNKED_PERSISTENT_QUALIFICATION_20261029.md',
              'docs/CHUNKED_PERSISTENT_QUALIFICATION_RECOVERY_20261029.md','tests/test_chunked_persistent_recovery.py'):
        files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);phases=[ID+'-warm',ID+'-replay']
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:
        raise RuntimeError('Two recovery snapshots plus inode reserve required')
    p.update(source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,phases),phases={n:{} for n in phases},inode_projection=projection,
        original_plan_sha256=file_sha256(oldroot/'plan.json'),original_failure_sha256=file_sha256(oldroot/'report.json'),
        original_failed_updates=len(updates),original_warm_times=times,
        original_cost_sha256=file_sha256(original.PUBLIC/'execution-costs.json'))
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def run(root):
    original.ID=ID;original.PUBLIC=PUBLIC
    original.run(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run'],required=True);a=p.parse_args()
    (prepare if a.action=='prepare' else run)(a.root)
