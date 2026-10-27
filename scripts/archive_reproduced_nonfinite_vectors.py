"""Exact-byte retention of two closed nonfinite vectors, never source data."""
import argparse,json,shutil,time
from pathlib import Path
from compute_trainium.exact_gzip_archive import create_archive,restore_archive,sha256_file


def main(root):
    folder=root/'scale-reproduced-nonfinite-exact-archive-v1';parent=root/'scale-reproduced-point-probe-v1'
    if folder.exists():raise FileExistsError('Preserve every archival attempt')
    report=json.loads((parent/'report.json').read_text())
    if report['status']!='COMPLETE_REPRODUCED_POINT_SCALE_DIAGNOSIS':raise ValueError('Only complete closed point evidence')
    files=[]
    for variant in ['high','low']:
        row=report['records'][variant];path=parent/(variant+'-gradient.pt')
        if row['gradient_finite'] or not row['parameters_unchanged'] or row['optimizer_updates']!=0:raise ValueError('Only these two observed nonfinite zero-update vectors')
        closure=root/'snapshots'/('scale-reproduced-point-probe-v1-'+variant)/'closure.json'
        if json.loads(closure.read_text())['exit_code']!=0:raise ValueError('Complete owned closure required')
        if path.is_symlink() or sha256_file(path)!=row['gradient_sha256']:raise AssertionError('Unchanged original regular vector required')
        files.append(dict(file=str(path.relative_to(root)),sha256=row['gradient_sha256'],bytes=path.stat().st_size,closure_sha256=sha256_file(closure)))
    if sum(f['bytes'] for f in files)>3*1024**3:raise ValueError('Fixed3GiB original scope')
    started=time.time();deadline=started+900
    folder.mkdir();(folder/'plan.json').write_text(json.dumps(dict(registered_unix=started,absolute_deadline_unix=deadline,
        files=files,source_sha256=sha256_file(Path(__file__)),assigned_chips=0,optimizer_updates=0,
        scope='Only two terminal nonfinite vector binaries. Every original byte/hash retained in verified lossless archives,including an actual restoration. No source/model/optimizer/quality/history changes.'),indent=2)+'\n')
    rows=[];result=dict(status='STOPPED_EXACT_NONFINITE_ARCHIVE',assigned_chips=0,optimizer_updates=0,original_failure_unchanged=True)
    try:
        for entry in files:
            source=root/entry['file'];needed=2*entry['bytes']+64*1024**2
            if time.time()>deadline:raise TimeoutError('Original15-minute archival cap')
            if shutil.disk_usage(root).free<35*1024**3+needed:raise RuntimeError('Full temporary copy plus protected35GiB')
            available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            if available<512*1024**3+256*1024**2:raise RuntimeError('Protected host reserve plus streaming workspace')
            recipe,row=create_archive(source)
            check=folder/('restored-'+source.name);restore_archive(recipe,check)
            if sha256_file(check)!=entry['sha256'] or sha256_file(source)!=entry['sha256']:raise AssertionError('Actual restored bytes differ')
            check.unlink();source.unlink();rows.append(dict(file=entry['file'],**row))
            (folder/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        result['status']='COMPLETE_EXACT_NONFINITE_VECTOR_ARCHIVE'
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(records=rows,wall_seconds=time.time()-started,all_completed_original_sha256_restorable=True)
        (folder/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
