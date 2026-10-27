"""Finish an interrupted cold archive as verified gzip segments, preserving it."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from compute_trainium.execution_isolation import managed_running
from compute_trainium.training import file_sha256

IDENTITY='workspace-discovery-capacity-preservation-v2'
PUBLIC=Path('results/workspace_discovery_storage_20261027/recovery-v2')


def run(root):
    if managed_running():raise RuntimeError('No Native worker may overlap this recovery')
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every archival attempt')
    prior=root/'workspace-discovery-capacity-preservation-v1';old=json.loads((prior/'report.json').read_text())
    if old['status']!='STOPPED_LEGACY_PACKING_PRESERVATION' or len(old['files'])!=1:raise AssertionError('Exact stopped first-file boundary required')
    registration=json.loads((prior/'plan.json').read_text());rows=registration['files'][1:]
    partial=prior/'1.gz.incomplete'
    if not partial.is_file():raise AssertionError('Original stopped archive must remain')
    out.mkdir(mode=0o700);PUBLIC.mkdir(parents=True);started=time.time()
    plan=dict(registered_unix=started,source_sha256=file_sha256(__file__),prior_report_sha256=file_sha256(prior/'report.json'),
        retained_partial_sha256=file_sha256(partial),files=rows,seconds_limit=2400,
        scope='Finish original stopped gzip as immutable prefix plus new tail;validate original whole-file bytes across both segments before release.Then archive remaining registered cold arrays.No states/source views/raw sources removed.41GiB archive floor reserves35GiB plus a possible6GiB CPU-audit diagnostic.')
    for d in [out,PUBLIC]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_WORKSPACE_CAPACITY_RECOVERY',files=[],optimizer_updates=0)
    def guard():
        if managed_running() or shutil.disk_usage(root).free<41*1024**3+32*1024**2 or time.time()-started>2400:raise RuntimeError('Whole recovery resource/time reserve')
    def read_segments(parts):
        digest=hashlib.sha256();size=0
        for archive in parts:
            with gzip.open(archive,'rb') as source:
                while data:=source.read(8*1024**2):guard();digest.update(data);size+=len(data)
        return digest.hexdigest(),size
    try:
        for index,row in enumerate(rows,1):
            guard();path=root/row['relative']
            if file_sha256(path)!=row['sha256'] or path.stat().st_mtime_ns!=row['mtime_ns']:raise AssertionError('Frozen original changed')
            offset=0;parts=[]
            if index==1:
                # The exception closed the original gzip stream normally. Its
                # complete decoded prefix is compared directly to the original.
                with gzip.open(partial,'rb') as decoded,path.open('rb') as original:
                    while data:=decoded.read(8*1024**2):
                        guard()
                        if original.read(len(data))!=data:raise AssertionError('Stopped gzip prefix differs from original')
                        offset+=len(data)
                if not 0<offset<row['bytes']:raise AssertionError('Expected incomplete original prefix')
                parts.append(partial)
            destination=out/(str(index)+'.tail.gz' if index==1 else str(index)+'.gz')
            temporary=destination.with_suffix(destination.suffix+'.incomplete')
            with temporary.open('xb') as raw:
                with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=1,mtime=0) as encoded:
                    with path.open('rb') as source:
                        source.seek(offset)
                        while data:=source.read(8*1024**2):guard();encoded.write(data)
            temporary.rename(destination);parts.append(destination)
            if read_segments(parts)!=(row['sha256'],row['bytes']):raise AssertionError('Whole segmented original roundtrip failed')
            if file_sha256(path)!=row['sha256'] or path.stat().st_mtime_ns!=row['mtime_ns']:raise AssertionError('Original changed before release')
            entry=dict(row,parts=[dict(relative=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=file_sha256(p)) for p in parts],
                recovered_prefix_bytes=offset,whole_original_roundtrip_verified=True)
            (out/(str(index)+'-verified.json')).write_text(json.dumps(entry,indent=2)+'\n')
            path.unlink();result['files'].append(entry)
            (out/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        result.update(status='COMPLETE_EXACT_WORKSPACE_CAPACITY_RECOVERY',recovered_bytes=sum(r['bytes']-sum(p['bytes'] for p in r['parts']) for r in result['files']))
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,free_bytes=shutil.disk_usage(root).free,plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,PUBLIC]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


def restore(root,relative):
    if managed_running():raise RuntimeError('Explicit cold restore only outside Native phases')
    report=json.loads((root/IDENTITY/'report.json').read_text());row=next(r for r in report['files'] if r['relative']==relative)
    path=root/relative
    if path.exists():raise FileExistsError('Never overwrite an original')
    if shutil.disk_usage(root).free<35*1024**3+row['bytes']:raise RuntimeError('Complete materialized restore allowance required')
    temporary=path.with_suffix('.npy.restore-incomplete')
    with temporary.open('xb') as writer:
        for part in row['parts']:
            archive=root/part['relative']
            if file_sha256(archive)!=part['sha256']:raise AssertionError('Preserved segment changed')
            with gzip.open(archive,'rb') as source:shutil.copyfileobj(source,writer,8*1024**2)
    if file_sha256(temporary)!=row['sha256']:raise AssertionError('Materialized restored original differs')
    os.chmod(temporary,row['mode']);os.utime(temporary,ns=(row['mtime_ns'],row['mtime_ns']));temporary.rename(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--restore');a=p.parse_args()
    restore(a.root,a.restore) if a.restore else run(a.root)
