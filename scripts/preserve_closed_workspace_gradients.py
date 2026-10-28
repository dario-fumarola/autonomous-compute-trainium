"""Exact cold diagnostic references;protect the live four-condition endpoints."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from compute_trainium.execution_isolation import managed_running,metadata
from compute_trainium.training import file_sha256

IDENTITY='workspace-closed-gradient-preservation-v1'
PUBLIC=Path('results/workspace_closed_gradients_20261027')


def run(root,*,more_cold=False):
    out=root/IDENTITY
    if out.exists() or PUBLIC.exists():raise FileExistsError('Never overwrite preservation evidence')
    parent=root/'workspace-loss-stability-diagnosis-v1'
    receipt=json.loads((parent/'report.json').read_text());rows=[]
    entries=([root/'workspace-identity-gradient-v1/gradients.pt',root/'workspace-stable-references-v1/c0-gradients.pt',root/'workspace-online-mature-cpu-audit-v1/failed-independent-gradients.pt'] if more_cold else [parent/(name+'-gradients.pt') for name in ['fp32','fp64']])
    for index,path in enumerate(entries):
        expected=file_sha256(path) if more_cold else receipt['gradient_sha256'][['fp32','fp64'][index]]
        if file_sha256(path)!=expected:raise AssertionError('Frozen diagnostic reference changed')
        rows.append(dict(relative=str(path.relative_to(root)),sha256=expected,bytes=path.stat().st_size))
    out.mkdir(mode=0o700);PUBLIC.mkdir();started=time.time();last=0.
    def guard(extra=0):
        nonlocal last
        if shutil.disk_usage(root).free<65*1024**3+extra:raise RuntimeError('35GiB floor plus complete30GiB active endpoint allowance')
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        if available<512*1024**3 or time.time()-started>1800:raise RuntimeError('Protected host/time reserve')
        if time.monotonic()-last>2:
            for container in managed_running():
                if not metadata(container)['Config']['Labels'].get('ac-trainium-owner','').startswith('workspace-four-arm-discovery-v'):raise RuntimeError('Unexpected active owner')
            last=time.monotonic()
    plan=dict(registered_unix=time.time(),source_sha256=file_sha256(__file__),files=rows,source_report_sha256=file_sha256(parent/'report.json'),
        scope='Only listed CLOSED correctness gradient references.No model,optimizer,active gradient,source view or LIVE160-position initial reference removed.pigz8threads;whole original hashes and full decoded readback before release.First complete materialized restore also checked.35GiB floor+30GiB next Native endpoint protected;maintenance overlaps the current bounded diagnostic and is not a speed comparison.')
    for d in [out,PUBLIC]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_CLOSED_GRADIENT_PRESERVATION',files=[],optimizer_updates=0)
    try:
        for index,row in enumerate(rows):
            guard(row['bytes']*(2 if index==0 else 1)+128*1024**2);path=root/row['relative'];archive=out/(str(index)+'.gz')
            with archive.open('xb') as writer:
                process=subprocess.Popen(['pigz','-1','-p','8','-n','-c',str(path)],stdout=writer,stderr=subprocess.PIPE)
                try:
                    while process.poll() is None:guard();time.sleep(.5)
                    if process.returncode:raise RuntimeError('Bounded pigz failed')
                except BaseException:
                    process.terminate();process.wait();raise
            h=hashlib.sha256();size=0;trial=out/'materialized-reference.pt';writer=trial.open('xb') if index==0 else None
            try:
                with gzip.open(archive,'rb') as source:
                    while data:=source.read(8*1024**2):
                        guard();h.update(data);size+=len(data)
                        if writer:writer.write(data)
            finally:
                if writer:writer.close()
            if h.hexdigest()!=row['sha256'] or size!=row['bytes']:raise AssertionError('Complete original readback differs')
            if index==0:
                if file_sha256(trial)!=row['sha256']:raise AssertionError('Materialized restore differs')
                trial.unlink()
            if file_sha256(path)!=row['sha256']:raise AssertionError('Original changed before release')
            entry=dict(row,archive_relative=str(archive.relative_to(root)),archive_sha256=file_sha256(archive),archive_bytes=archive.stat().st_size,whole_original_roundtrip_verified=True,materialized_restore_verified=index==0)
            (out/(str(index)+'-verified.json')).write_text(json.dumps(entry,indent=2)+'\n');path.unlink();result['files'].append(entry)
        result.update(status='COMPLETE_EXACT_CLOSED_GRADIENT_PRESERVATION',recovered_bytes=sum(r['bytes']-r['archive_bytes'] for r in result['files']))
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,PUBLIC]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


def restore(root,relative):
    receipt=json.loads((root/IDENTITY/'report.json').read_text());row=next(r for r in receipt['files'] if r['relative']==relative)
    path=root/relative;archive=root/row['archive_relative']
    if managed_running() or path.exists():raise RuntimeError('Explicit non-overwriting restore outside Native phases only')
    if shutil.disk_usage(root).free<35*1024**3+row['bytes'] or file_sha256(archive)!=row['archive_sha256']:raise RuntimeError('Complete intact restore allowance required')
    temporary=path.with_suffix('.pt.restore-incomplete')
    with gzip.open(archive,'rb') as source,temporary.open('xb') as writer:shutil.copyfileobj(source,writer,8*1024**2)
    if file_sha256(temporary)!=row['sha256']:raise AssertionError('Restored original differs')
    temporary.rename(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--restore');p.add_argument('--more-cold',action='store_true');a=p.parse_args()
    if a.more_cold:
        IDENTITY='workspace-closed-c0-gradient-preservation-v2';PUBLIC=Path('results/workspace_closed_gradients_20261027/recovery-v2')
    restore(a.root,a.restore) if a.restore else run(a.root,more_cold=a.more_cold)
