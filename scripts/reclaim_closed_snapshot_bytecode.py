"""Reclaim reproducible Python caches, retaining every closed source snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def source_for_cache(snapshot,path,manifest):
    if (snapshot.is_symlink() or path.is_symlink() or path.parent.is_symlink()
            or path.parent.name!='__pycache__' or not path.name.endswith('.pyc')
            or '.cpython-' not in path.name):return None
    try:path.resolve().relative_to(snapshot.resolve())
    except ValueError:return None
    source=path.parent.parent/(path.name.split('.cpython-',1)[0]+'.py')
    relative=str(source.relative_to(snapshot))
    if not source.is_file() or source.is_symlink() or relative not in manifest:return None
    if file_sha256(source)!=manifest[relative]:raise AssertionError('Recorded source changed; preserve its cache for diagnosis')
    return source


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--apply',action='store_true');p.add_argument('--limit',type=int,default=30000);a=p.parse_args();started=time.time()
    if not 1<=a.limit<=30000:raise ValueError('Bounded cache-only reclamation required')
    if a.output.exists():raise FileExistsError('Preserve cache maintenance receipt')
    if managed_running():raise RuntimeError('Closed-snapshot maintenance requires no native owners')
    candidates=[];snapshots={};before=os.statvfs(a.root).f_favail
    for snapshot in sorted((a.root/'snapshots').iterdir()):
        if not snapshot.is_dir() or snapshot.is_symlink() or not (snapshot/'closure.json').exists():continue
        launch=snapshot/'launch.json'
        if not launch.exists():continue
        manifest=json.loads(launch.read_text())['source_sha256']
        for path in sorted(snapshot.rglob('*.pyc')):
            if len(candidates)>=a.limit:break
            source=source_for_cache(snapshot,path,manifest)
            if source:
                candidates.append((snapshot,path,source,path.stat().st_size))
                snapshots.setdefault(snapshot,dict(launch_sha256=file_sha256(launch),closure_sha256=file_sha256(snapshot/'closure.json'),sources={}))['sources'][str(source.relative_to(snapshot))]=manifest[str(source.relative_to(snapshot))]
        if len(candidates)>=a.limit:break
    if managed_running():raise RuntimeError('A native owner appeared before maintenance')
    removed=0;total=0
    if a.apply:
        for snapshot,path,source,size in candidates:
            if time.time()-started>180:raise TimeoutError('Bounded cache maintenance cap')
            if file_sha256(source)!=snapshots[snapshot]['sources'][str(source.relative_to(snapshot))]:raise AssertionError('Preserved source changed before unlink')
            path.unlink();removed+=1;total+=size
        for directory in {path.parent for _,path,_,_ in candidates}:
            if not any(directory.iterdir()):directory.rmdir()
    verified=0
    for snapshot,record in snapshots.items():
        if file_sha256(snapshot/'launch.json')!=record['launch_sha256'] or file_sha256(snapshot/'closure.json')!=record['closure_sha256']:
            raise AssertionError('Closed execution receipts changed')
        for source,sha in record['sources'].items():
            if file_sha256(snapshot/source)!=sha:raise AssertionError('Maintenance changed a frozen source file')
            verified+=1
    result=dict(status='COMPLETE_CLOSED_BYTECODE_RECLAMATION' if a.apply else 'DRY_RUN_CLOSED_BYTECODE_RECLAMATION',
        source_sha256=file_sha256(__file__),eligible_cache_files=len(candidates),removed_cache_files=removed,
        removed_cache_bytes=total,closed_snapshots=len(snapshots),verified_source_files=verified,
        free_inodes_before=before,free_inodes_after=os.statvfs(a.root).f_favail,wall_seconds=time.time()-started,
        model_or_optimizer_files_touched=0,source_files_removed=0,assigned_chips=0,test_accessed=False,
        scope='Only CPython bytecode with a present matching frozen source hash,inside terminal source snapshots. Every referenced source and launch/closure receipt verified unchanged. Models,optimizer states,data,NEFF/HLO compiler caches and live source trees are untouched. Python bytecode can be regenerated from the retained sources.')
    a.output.parent.mkdir(parents=True,exist_ok=True);write_state(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
