"""Preserve every byte/path while sharing identical recent closed snapshots."""
import argparse
import concurrent.futures as futures
import json
import os
from pathlib import Path
import shutil
import time

from compute_trainium.closed_source_storage import source_record,source_actions,share_source
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID='recent-closed-source-sharing-20261030-v1'
PUBLIC=Path('results/closed_source_inode_sharing_20261030/recent-v1')


def main(root,apply=False):
    start=time.time();out=root/ID
    read=lambda p:json.loads(p.read_text())
    def guard(deadline):
        if managed_running() or time.time()>deadline:raise RuntimeError('Closed owner boundary and original ten-minute cap')
        if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Storage reserve')
    guard(start+600)
    if not apply:
        if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every inventory attempt')
        out.mkdir();PUBLIC.mkdir(parents=True)
        registration=dict(identity=ID,registered_unix=start,absolute_deadline_unix=start+600,
            maximum_snapshots=24,maximum_source_paths=30000,source_sha256={f:file_sha256(f) for f in
                ('scripts/share_recent_closed_sources.py','compute_trainium/closed_source_storage.py','tests/test_closed_source_storage.py')},
            scope='Only the24 most recently closed snapshots at registration. Manifest-listed byte/permission-identical Python sources share storage atomically. Preserve every original source path, bytes and launch/closure receipt. Exclude live worktrees, models, optimizer states, source data and compiler caches. Historical aliases remain immutable; copy before debugging.')
        write_state(out/'registration.json',registration);write_state(PUBLIC/'registration.json',registration)
        result=dict(status='STOPPED_RECENT_SOURCE_INVENTORY',source_mutations=0)
        try:
            eligible=[p for p in (root/'snapshots').iterdir() if not p.is_symlink()
                and (p/'launch.json').is_file() and (p/'closure.json').is_file()]
            selected=sorted(eligible,key=lambda p:((p/'closure.json').stat().st_mtime_ns,p.name),reverse=True)[:24]
            jobs=[];snapshots={}
            for snapshot in selected:
                guard(registration['absolute_deadline_unix'])
                launch=read(snapshot/'launch.json')
                snapshots[snapshot.name]={k+'_sha256':file_sha256(snapshot/(k+'.json')) for k in ('launch','closure')}
                jobs.extend((snapshot,n,h) for n,h in launch['source_sha256'].items())
            if len(jobs)>30000:raise RuntimeError('Bounded recent inventory exceeded')
            with futures.ThreadPoolExecutor(8) as pool:
                records=list(pool.map(lambda a:source_record(root,*a),jobs))
            actions=source_actions(records);plan=dict(records=records,actions=actions,snapshots=snapshots)
            write_state(out/'plan.json',plan)
            result.update(status='COMPLETE_RECENT_SOURCE_INVENTORY',source_paths=len(records),closed_snapshots=len(snapshots),
                candidate_aliases=len(actions),conservative_reclaimable_inodes=len({a['duplicate']['inode'] for a in actions if a['duplicate']['links']==1}),
                plan_sha256=file_sha256(out/'plan.json'))
        except BaseException as error:
            result.update(error_type=type(error).__name__,error=str(error));raise
        finally:
            result.update(wall_seconds=time.time()-start,assigned_chips=0)
            write_state(out/'inventory.json',result);write_state(PUBLIC/'inventory.json',result);print(json.dumps(result),flush=True)
        return
    registration=read(out/'registration.json');guard(registration['absolute_deadline_unix'])
    if any(file_sha256(f)!=h for f,h in registration['source_sha256'].items()):raise AssertionError('Storage operator changed')
    inventory=read(out/'inventory.json')
    if inventory['status']!='COMPLETE_RECENT_SOURCE_INVENTORY' or file_sha256(out/'plan.json')!=inventory['plan_sha256']:
        raise AssertionError('Complete exact inventory required')
    if (out/'started.json').exists():raise FileExistsError('No automatic partial retry')
    plan=read(out/'plan.json');before=os.statvfs(root).f_favail;completed=0
    write_state(out/'started.json',dict(unix=start));result=dict(status='STOPPED_RECENT_SOURCE_SHARING')
    try:
        with (out/'PRIVATE-completed-paths.jsonl').open('x') as journal, futures.ThreadPoolExecutor(8) as pool:
            for left in range(0,len(plan['actions']),256):
                guard(registration['absolute_deadline_unix'])
                for path in pool.map(lambda a:share_source(root,a),plan['actions'][left:left+256]):
                    journal.write(json.dumps(path)+'\n');completed+=1
                journal.flush()
        def verify(row):
            if file_sha256(root/row['path'])!=row['sha256']:raise AssertionError('Frozen source changed')
            return True
        with futures.ThreadPoolExecutor(8) as pool:verified=sum(pool.map(verify,plan['records']))
        for name,record in plan['snapshots'].items():
            for kind in ('launch','closure'):
                if file_sha256(root/'snapshots'/name/(kind+'.json'))!=record[kind+'_sha256']:
                    raise AssertionError('Execution receipt changed')
        result.update(status='COMPLETE_RECENT_BYTE_PRESERVING_SOURCE_SHARING',verified_source_paths=verified,
            all_launch_and_closure_receipts_unchanged=True)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(verified_alias_replacements=completed,free_inodes_before=before,
            free_inodes_after=os.statvfs(root).f_favail,wall_seconds=time.time()-start,
            original_source_bytes_changed=0,model_or_corpus_or_checkpoint_files_modified=0,
            assigned_chips=0,optimizer_updates=0,plan_sha256=inventory['plan_sha256'],scope=registration['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--apply',action='store_true')
    a=p.parse_args();main(a.root,a.apply)
