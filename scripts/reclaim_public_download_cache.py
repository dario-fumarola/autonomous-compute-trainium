"""Evict only pinned public download caches;preserve accepted packed corpora."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import quote

import requests
from compute_trainium.execution_isolation import managed_running
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state

IDENTITY='workspace-public-download-cache-eviction-v1'
PUBLIC=Path('results/workspace_public_cache_20261027')
GIB=1024**3


def url(row):
    receipt=row['receipt']
    return 'https://huggingface.co/datasets/'+receipt['repository']+'/resolve/'+receipt['revision']+'/'+quote(receipt['file'],safe='/')


def remote_sha(row):
    response=requests.head(url(row),allow_redirects=False,timeout=30)
    actual=response.headers.get('x-linked-etag',response.headers.get('etag','')).strip('"')
    if response.status_code not in [200,302,307] or actual!=row['receipt']['sha256']:
        raise ValueError('Pinned remote cache fingerprint unavailable;do not evict '+row['file'])
    return actual


def download(row,target):
    if target.exists():raise FileExistsError('Never overwrite any original or partial download')
    size=0;digest=hashlib.sha256()
    if shutil.disk_usage(target.parent).free<35*GIB+row['bytes']+64*1024**2:raise RuntimeError('Complete redownload peak plus reserve required')
    with requests.get(url(row),stream=True,timeout=(30,30)) as response:
        if response.status_code!=200:raise RuntimeError('Pinned download HTTP '+str(response.status_code))
        with target.open('xb') as output:
            for block in response.iter_content(4*1024**2):
                output.write(block);digest.update(block);size+=len(block)
                if size>row['bytes'] or shutil.disk_usage(target.parent).free<35*GIB:raise RuntimeError('Bounded download or protected reserve exceeded')
    if size!=row['bytes'] or digest.hexdigest()!=row['receipt']['sha256']:raise AssertionError('Actual redownload differs from original bytes')


def main(root,apply=False,restore=None):
    out=root/IDENTITY;source=root/'natural-append-capacity-v4/raw-source-map.json'
    if restore:
        plan=json.loads((out/'plan.json').read_text());row=next(r for r in plan['files'] if r['file']==restore)
        target=root/row['file']
        if target.exists():
            if file_sha256(target)!=row['receipt']['sha256']:raise AssertionError('Existing cache differs')
            return
        temporary=target.with_name(target.name+'.restore-incomplete');download(row,temporary)
        os.link(temporary,target);temporary.unlink();target.chmod(row['mode'])
        os.utime(target,ns=(row['mtime_ns'],row['mtime_ns']));return
    with (root/'native-half-reservations.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if managed_running():raise RuntimeError('No cache maintenance over Native workers')
        if not (out/'plan.json').exists():
            if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every cache maintenance attempt')
            files=[];seen=set()
            for record in json.loads(source.read_text()):
                p=Path(record['path']);receipt=record['receipt']
                if (not p.is_relative_to(root) or not p.is_file() or p.is_symlink() or p in seen
                        or p.suffix not in ['.parquet','.gz']):continue
                if len(receipt['revision'])!=40 or len(receipt['sha256'])!=64:raise ValueError('Immutable public revisions and byte hashes required')
                seen.add(p);s=p.stat();files.append(dict(file=str(p.relative_to(root)),receipt=receipt,bytes=s.st_size,
                    inode=s.st_ino,mode=s.st_mode&0o777,mtime_ns=s.st_mtime_ns))
            if not files or sum(r['bytes'] for r in files)>70*GIB:raise ValueError('Bounded at-most70GiB public-cache scope required')
            with ThreadPoolExecutor(4) as pool:list(pool.map(remote_sha,files))
            out.mkdir();PUBLIC.mkdir();plan=dict(identity=IDENTITY,registered_unix=time.time(),
                source_sha256=file_sha256(__file__),raw_map_sha256=file_sha256(source),files=files,
                maximum_bytes=70*GIB,target_free_bytes=105*GIB,maximum_seconds=900,assigned_chips=0,optimizer_updates=0,
                scope='Only manifest-listed public HF download caches inside this task root. Remote immutable revision and original content SHA verified. First complete byte-identical redownload required before eviction. All accepted packed/source-token arrays,source descriptors,private inputs,weights,checkpoints and receipts remain. The three currently used source caches outside this root are excluded. Restore exact cache paths with --restore-file.')
            write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
        if not apply:return
        plan=json.loads((out/'plan.json').read_text())
        if (out/'report.json').exists():raise FileExistsError('Preserve every completed or stopped eviction')
        if file_sha256(__file__)!=plan['source_sha256'] or file_sha256(source)!=plan['raw_map_sha256']:raise AssertionError('Cache operator or descriptors changed')
        started=time.time();records=[];result=dict(status='STOPPED_PUBLIC_CACHE_EVICTION',assigned_chips=0,optimizer_updates=0)
        try:
            sample=min(plan['files'],key=lambda row:row['bytes']);temporary=out/'verified-redownload.bin'
            download(sample,temporary);temporary.unlink();result['actual_original_redownload_verified']=sample['receipt']['sha256']
            for row in sorted(plan['files'],key=lambda r:(-r['bytes'],r['file'])):
                if shutil.disk_usage(root).free>=plan['target_free_bytes']:break
                if time.time()-started>900 or managed_running():raise RuntimeError('Exclusive cache allocation or watchdog ended')
                available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
                if available<512*GIB:raise RuntimeError('Protected host reserve')
                file=root/row['file'];s=file.stat()
                if file.is_symlink() or (s.st_ino,s.st_size)!=(row['inode'],row['bytes']):raise AssertionError('Original cache identity changed')
                remote_sha(row)
                if file_sha256(file)!=row['receipt']['sha256']:raise AssertionError('Local original cache differs;preserve it')
                file.unlink();records.append(row['file']);write_state(out/'progress.json',dict(files=len(records),free_bytes=shutil.disk_usage(root).free))
            result['status']='COMPLETE_BOUNDED_PUBLIC_CACHE_EVICTION'
        except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
        finally:
            result.update(evicted_files=records,files=len(records),wall_seconds=time.time()-started,
                recovered_bytes=sum(r['bytes'] for r in plan['files'] if r['file'] in records),actual_root_free_after=shutil.disk_usage(root).free,
                accepted_packed_corpora_unchanged=True,all_original_download_receipts_retained=True)
            write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--apply',action='store_true')
    p.add_argument('--restore-file');a=p.parse_args();main(a.root,apply=a.apply,restore=a.restore_file)
