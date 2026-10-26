"""Bounded preservation of all closed-source paths with fewer duplicate inodes."""
import argparse
import concurrent.futures as futures
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time
import traceback

from compute_trainium.closed_source_storage import source_record,source_actions,share_source
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='closed-source-inode-sharing-v1'
PUBLIC=Path('results/closed_source_inode_sharing_20261026')


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    out=a.root/IDENTITY;started=time.time();cpu=time.process_time()
    if managed_running():raise RuntimeError('No storage maintenance while native owners are live')
    def guard():
        if time.time()-started>900 or min(shutil.disk_usage(a.root).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Bounded source-preservation cap/reserve')
        if managed_running():raise RuntimeError('Unexpected new native owner')
    plan_file=out/'plan.json'
    if not plan_file.exists():
        if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve stopped source inventory')
        out.mkdir();PUBLIC.mkdir(parents=True)
        files=['scripts/share_closed_source_snapshots.py','compute_trainium/closed_source_storage.py','tests/test_closed_source_storage.py']
        write_state(out/'registration.json',dict(identity=IDENTITY,registered_unix=started,seconds_limit=900,
            source_sha256={f:file_sha256(f) for f in files},maximum_closed_source_paths=250000,
            array_or_source_content_rewritten=False,scope='Read-only inventory,then separately executed atomic hard links between byte-identical manifest-listed .py files in CLOSED source snapshots only. Preserve all file paths,byte hashes,launch/closure receipts,models,corpora and checkpoints. Original inode/mtime/owner metadata stays in the private inventory. Never edit these immutable historical source aliases;make a separate copy for intentional debugging. Current live worktree sources are excluded.'))
        write_state(PUBLIC/'registration.json',read(out/'registration.json'))
        jobs=[];snapshots={}
        for snapshot in sorted((a.root/'snapshots').iterdir()):
            if snapshot.is_symlink() or not (snapshot/'launch.json').is_file() or not (snapshot/'closure.json').is_file():continue
            guard();launch=read(snapshot/'launch.json')
            snapshots[snapshot.name]=dict(launch_sha256=file_sha256(snapshot/'launch.json'),closure_sha256=file_sha256(snapshot/'closure.json'))
            jobs.extend((snapshot,name,h) for name,h in launch['source_sha256'].items())
        if len(jobs)>250000:raise RuntimeError('Explicit source inventory bound exceeded')
        with futures.ThreadPoolExecutor(8) as pool:
            records=list(pool.map(lambda args:source_record(a.root,*args),jobs))
        actions=source_actions(records)
        plan=dict(records=records,actions=actions,snapshots=snapshots)
        write_state(plan_file,plan)
        report=dict(status='COMPLETE_READ_ONLY_FROZEN_SOURCE_INVENTORY',source_paths=len(records),closed_snapshots=len(snapshots),
            candidate_aliases=len(actions),conservative_reclaimable_inodes=len({v['duplicate']['inode'] for v in actions if v['duplicate']['links']==1}),
            plan_sha256=file_sha256(plan_file),wall_seconds=time.time()-started,source_mutations=0,assigned_chips=0)
        write_state(out/'inventory-report.json',report);write_state(PUBLIC/'inventory-report.json',report)
        print(json.dumps(report),flush=True)
    if a.prepare_only:return
    registration=read(out/'registration.json')
    if any(file_sha256(f)!=h for f,h in registration['source_sha256'].items()):raise AssertionError('Verified storage operator changed')
    if (out/'started.json').exists():raise FileExistsError('Preserve each interrupted or completed sharing operation')
    inventory=read(out/'inventory-report.json')
    if file_sha256(plan_file)!=inventory['plan_sha256']:raise AssertionError('Exact closed-source inventory changed')
    plan=read(plan_file);before=os.statvfs(a.root).f_favail;completed=0;result=dict(status='STOPPED_CLOSED_SOURCE_INODE_SHARING')
    write_state(out/'started.json',dict(unix=time.time(),inventory_plan_sha256=inventory['plan_sha256']))
    def interrupted(number,frame):raise KeyboardInterrupt('Bounded source sharing interrupted: '+str(number))
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        with (out/'completed-paths.jsonl').open('x') as journal, futures.ThreadPoolExecutor(8) as pool:
            for left in range(0,len(plan['actions']),512):
                guard()
                for path in pool.map(lambda action:share_source(a.root,action),plan['actions'][left:left+512]):
                    journal.write(json.dumps(path)+'\n');completed+=1
                journal.flush()
                write_state(out/'progress.json',dict(verified_shared_paths=completed,wall_seconds=time.time()-started))
        def verify(row):
            if file_sha256(a.root/row['path'])!=row['sha256']:raise AssertionError('An original frozen source path changed')
            return True
        with futures.ThreadPoolExecutor(8) as pool:verified=sum(pool.map(verify,plan['records']))
        for name,receipt in plan['snapshots'].items():
            for label in ['launch','closure']:
                if file_sha256(a.root/'snapshots'/name/(label+'.json'))!=receipt[label+'_sha256']:
                    raise AssertionError('Historical execution receipt changed')
        result.update(status='COMPLETE_BYTE_PRESERVING_CLOSED_SOURCE_SHARING',all_original_source_paths_verified=verified,
            all_launch_and_closure_receipts_unchanged=True,closed_snapshots=len(plan['snapshots']))
    except BaseException as error:
        result.update(error=str(error),error_kind=type(error).__name__,traceback=traceback.format_exc());raise
    finally:
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(verified_alias_replacements=completed,free_inodes_before=before,free_inodes_after=os.statvfs(a.root).f_favail,
            wall_seconds=time.time()-started,cpu_seconds=time.process_time()-cpu,
            original_source_bytes_changed=0,model_or_corpus_or_checkpoint_files_modified=0,
            model_updates=0,assigned_chips=0,plan_sha256=inventory['plan_sha256'],scope=registration['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
