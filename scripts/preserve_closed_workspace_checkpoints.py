"""Byte-exact cold checkpoint archives;never touch active discovery endpoints."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from compute_trainium.execution_isolation import managed_running,metadata
from compute_trainium.training import file_sha256

IDENTITY='workspace-closed-checkpoint-preservation-v1'
PUBLIC=Path('results/workspace_closed_checkpoints_20261028')


def run(root):
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Retain every archival attempt')
    entries=[root/'runs'/parent/f'rank{rank}-u{update}.zip' for parent,update in [
        ('workspace-native-online-throughput-v1-online',102),('workspace-native-full-update-v1-gradient',1)] for rank in range(64)]
    rows=[dict(relative=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=file_sha256(p)) for p in entries]
    out.mkdir(mode=0o700);PUBLIC.mkdir();started=time.time();last=0.
    def guard():
        nonlocal last
        # Includes final workspace save while its rolling backup still exists.
        if shutil.disk_usage(root).free<65*1024**3:raise RuntimeError('35GiB floor plus30GiB active write allowance')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<512*1024**3 or time.time()-started>1800:raise RuntimeError('Protected host/time reserve')
        if time.monotonic()-last>2:
            for container in managed_running():
                if not metadata(container)['Config']['Labels'].get('ac-trainium-owner','').startswith('workspace-four-arm-discovery-v3'):raise RuntimeError('Unexpected live owner')
            last=time.monotonic()
    plan=dict(registered_unix=started,source_sha256=file_sha256(__file__),files=rows,
        scope='Only two CLOSED qualification histories;old failed states remain exactly restorable,including original ZIP bytes.All64 fresh qualifiedu101,currentC0 and every scientific endpoint/gradient remain raw.Whole gzip readback and a materialized restore precede release.Active200GiB worker keeps512GiB host reserve;35GiB disk plus30GiB active-write allowance.No optimizer updates or scientific data.')
    for d in [out,PUBLIC]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_CLOSED_CHECKPOINT_PRESERVATION',files=[],optimizer_updates=0)
    try:
        for index,row in enumerate(rows):
            guard();path=root/row['relative'];archive=out/(str(index)+'.gz')
            with archive.open('xb') as writer:
                child=subprocess.Popen(['pigz','-1','-p','8','-n','-c',str(path)],stdout=writer,stderr=subprocess.PIPE)
                try:
                    while child.poll() is None:guard();time.sleep(.2)
                    if child.returncode:raise RuntimeError('Closed-state compression failed')
                except BaseException:child.terminate();child.wait();raise
            digest=hashlib.sha256();size=0;trial=out/'restore-proof.zip';writer=trial.open('xb') if index==0 else None
            try:
                with gzip.open(archive,'rb') as decoded:
                    while chunk:=decoded.read(8*1024**2):
                        guard();digest.update(chunk);size+=len(chunk)
                        if writer:writer.write(chunk)
            finally:
                if writer:writer.close()
            if digest.hexdigest()!=row['sha256'] or size!=row['bytes']:raise AssertionError('Whole original ZIP roundtrip differs')
            if index==0:
                if file_sha256(trial)!=row['sha256']:raise AssertionError('Materialized original differs')
                trial.unlink()
            if file_sha256(path)!=row['sha256']:raise AssertionError('Closed checkpoint changed')
            entry=dict(row,archive_relative=str(archive.relative_to(root)),archive_bytes=archive.stat().st_size,archive_sha256=file_sha256(archive),whole_original_roundtrip_verified=True,materialized_restore_verified=index==0)
            (out/(str(index)+'-verified.json')).write_text(json.dumps(entry,indent=2)+'\n')
            path.unlink();result['files'].append(entry)
            (out/'progress.json').write_text(json.dumps(dict(files_completed=len(result['files']),recovered_bytes=sum(r['bytes']-r['archive_bytes'] for r in result['files']))))
        result['status']='COMPLETE_EXACT_CLOSED_CHECKPOINT_PRESERVATION'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(recovered_bytes=sum(r['bytes']-r['archive_bytes'] for r in result['files']),wall_seconds=time.time()-started,plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,PUBLIC]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


def restore(root,relative):
    receipt=json.loads((root/IDENTITY/'report.json').read_text());row=next(r for r in receipt['files'] if r['relative']==relative)
    path=root/relative;archive=root/row['archive_relative']
    if managed_running() or path.exists():raise RuntimeError('Explicit non-overwriting restore outside Native execution')
    if shutil.disk_usage(root).free<35*1024**3+row['bytes'] or file_sha256(archive)!=row['archive_sha256']:raise RuntimeError('Whole intact restore allowance required')
    temporary=path.with_suffix('.zip.restore-incomplete')
    with gzip.open(archive,'rb') as source,temporary.open('xb') as writer:shutil.copyfileobj(source,writer,8*1024**2)
    if file_sha256(temporary)!=row['sha256']:raise AssertionError('Restored original differs')
    temporary.rename(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--restore');a=p.parse_args()
    restore(a.root,a.restore) if a.restore else run(a.root)
