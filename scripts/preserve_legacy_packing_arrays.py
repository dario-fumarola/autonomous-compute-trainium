"""Exact cold-array compression;keeps Native final-state allowance untouched."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from compute_trainium.training import file_sha256
from compute_trainium.execution_isolation import managed_running,metadata

PUBLIC=Path('results/workspace_legacy_array_preservation_20261027')
IDENTITY='workspace-legacy-packing-preservation-v1'
NATIVE='workspace-native-online-throughput-v1-online'
RELATIVES=[
 'natural-append-3250m/extension/packed/domains.npy',
 'natural-append-3250m/packed/domains.npy',
 'natural-append-6500m-v2/extension/packed/domains.npy',
 'natural-append-6500m-v2/packed/domains.npy',
 'natural-append-capacity-v4/extension/packed/domains.npy',
 'natural-append-capacity-v4/packed/domains.npy',
 'natural-append-3250m/packed/tokens.npy',
 'natural-append-capacity-v4/extension/packed/tokens.npy']


def active_scope():
    for c in managed_running():
        m=metadata(c)
        if m['Config']['Labels'].get('ac-trainium-owner')!=NATIVE:raise RuntimeError('Only independent online qualification may overlap cold-array preservation')


def remaining_native_allowance(root):
    out=root/'runs'/NATIVE
    if not out.exists():return 30*1024**3
    written=sum(p.stat().st_size for p in out.iterdir() if p.is_file() and ('u102.zip' in p.name or 'mature-gradients.pt' in p.name))
    return max(0,30*1024**3-written)


def run(root):
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every archival attempt')
    active_scope();out.mkdir(mode=0o700);PUBLIC.mkdir()
    lock=(root/'legacy-packing-preservation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();rows=[]
    for relative in RELATIVES:
        path=root/relative;manifest=path.parent/'manifest.json';m=json.loads(manifest.read_text());stat=path.stat()
        rows.append(dict(relative=relative,sha256=m['files'][path.name],bytes=stat.st_size,mode=stat.st_mode&0o777,mtime_ns=stat.st_mtime_ns,
            archive_relative=IDENTITY+'/'+str(len(rows))+'.gz',manifest_sha256=file_sha256(manifest)))
    plan=dict(registered_unix=time.time(),files=rows,source_sha256=file_sha256(__file__),native_final_artifact_allowance_bytes=30*1024**3,
        scope='Only the listed cold legacy packing arrays. No raw source cache,accepted metadata/database,model,optimizer,gradient,source code or ZIP dependency parent is removed. Native online qualification reads only explicit checkpoint/input files;source-view extraction reads raw sources/databases,never these arrays. Whole-original SHA matched to frozen packing manifest,complete gzip readback plus first materialized restore before unlink. Original bytes remain exactly restorable;legacy mmap loaders must restore first.35GiB filesystem reserve plus remaining Native30GiB checkpoint/gradient allowance protected throughout.')
    for d in [out,PUBLIC]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def guard(extra=0):
        if shutil.disk_usage(root).free<35*1024**3+remaining_native_allowance(root)+extra:raise RuntimeError('Protected reserve plus complete remaining Native artifacts')
        if time.time()-started>1800:raise TimeoutError('Bounded lossless preservation')
    result=dict(status='STOPPED_LEGACY_PACKING_PRESERVATION',files=[],optimizer_updates=0)
    def decompress(archive,destination=None):
        h=hashlib.sha256();size=0;writer=destination.open('xb') if destination else None
        try:
            with gzip.open(archive,'rb') as source:
                while data:=source.read(8*1024**2):
                    guard(8*1024**2);h.update(data);size+=len(data)
                    if writer:writer.write(data)
        finally:
            if writer:writer.close()
        return h.hexdigest(),size
    try:
        for index,row in enumerate(rows):
            active_scope();path=root/row['relative'];archive=root/row['archive_relative'];temporary=archive.with_suffix('.gz.incomplete')
            # Full worst-case compressed output and first materialized restore.
            guard(row['bytes']+128*1024**2+(row['bytes'] if index==0 else 0))
            if file_sha256(path)!=row['sha256']:raise AssertionError('Frozen original packing bytes changed')
            digest=hashlib.sha256();written=0
            with temporary.open('xb') as raw:
                with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=1,mtime=0) as encoded:
                    with path.open('rb') as source:
                        while data:=source.read(8*1024**2):
                            guard(16*1024**2);digest.update(data);written+=len(data);encoded.write(data)
            if digest.hexdigest()!=row['sha256'] or written!=row['bytes']:raise AssertionError('Original changed during compression')
            if decompress(temporary)!=(row['sha256'],row['bytes']):raise AssertionError('Complete byte-exact decompression failed')
            if index==0:
                guard(row['bytes']+16*1024**2);trial=out/'materialized-restore.npy'
                if decompress(temporary,trial)!=(row['sha256'],row['bytes']) or file_sha256(trial)!=row['sha256']:raise AssertionError('Materialized restore differs')
                trial.unlink()
            if file_sha256(path)!=row['sha256'] or path.stat().st_mtime_ns!=row['mtime_ns']:raise AssertionError('Original changed before release')
            temporary.rename(archive);archive_sha=file_sha256(archive)
            receipt=dict(row,compressed_bytes=archive.stat().st_size,compressed_sha256=archive_sha,whole_original_roundtrip_verified=True,
                materialized_restore_verified=index==0)
            result['files'].append(receipt)
            # Durable receipt precedes release;original manifest stays untouched.
            (out/f'file-{index}-verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
            path.unlink();guard();(out/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        result.update(status='COMPLETE_EXACT_LEGACY_PACKING_PRESERVATION',recovered_bytes=sum(v['bytes']-v['compressed_bytes'] for v in result['files']))
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,free_bytes=shutil.disk_usage(root).free,plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,PUBLIC]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        lock.close()


def restore(root,relative):
    report=json.loads((root/IDENTITY/'report.json').read_text());row=next(v for v in report['files'] if v['relative']==relative)
    path=root/relative;archive=root/row['archive_relative']
    if path.exists():raise FileExistsError('No overwrite on exact restore')
    if shutil.disk_usage(root).free<35*1024**3+row['bytes']:raise RuntimeError('Full restore plus protected reserve required')
    if file_sha256(archive)!=row['compressed_sha256']:raise AssertionError('Preserved archive changed')
    temporary=path.with_suffix('.npy.restore-incomplete')
    with gzip.open(archive,'rb') as source,temporary.open('xb') as output:shutil.copyfileobj(source,output,8*1024**2)
    if file_sha256(temporary)!=row['sha256']:raise AssertionError('Restored bytes differ')
    os.chmod(temporary,row['mode']);os.utime(temporary,ns=(row['mtime_ns'],row['mtime_ns']));temporary.rename(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--restore');a=p.parse_args()
    restore(a.root,a.restore) if a.restore else run(a.root)
