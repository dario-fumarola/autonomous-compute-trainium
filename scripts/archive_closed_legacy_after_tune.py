"""One bounded inactive-instance archive to finish exact main-state admission."""
import json
from pathlib import Path
import shutil
import time

from compute_trainium.exact_gzip_archive import create_archive,restore_archive,sha256_file
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state

IDENTITY='closed-legacy-post-tune-exact-archive-v1'
GIB=1024**3


def archive(root,*,required_free_bytes,tuning_report,scope_file):
    if not isinstance(required_free_bytes,int) or required_free_bytes<=35*GIB:
        raise ValueError('Full positive post-reserve storage target required')
    out=root/IDENTITY
    if out.exists():raise FileExistsError('Preserve every inactive archive attempt')
    if managed_running():raise RuntimeError('Standalone preservation needs all Native workers closed')
    if (root/'scale-body-exposure-main-v1').exists():raise RuntimeError('No preservation over an admitted main')
    preceding=root/'closed-legacy-checkpoint-exact-archive-v3'
    if preceding.exists() and not (preceding/'report.json').exists():
        raise RuntimeError('The preceding archival stream must close first')
    tune_sha=sha256_file(tuning_report);scope_sha=sha256_file(scope_file)
    if json.loads(tuning_report.read_text())['status']!='COMPLETE_BOUNDARY_LOWER_INDEPENDENT_RATE_SELECTION':
        raise ValueError('Complete unchanged tuning required')
    protected=set()
    for p in root.glob('closed-legacy-zip-*/plan.json'):
        protected.update(json.loads(p.read_text()).get('raw_dependency_parents_must_remain',[]))
    candidates=[]
    for folder in (root/'runs').iterdir():
        if not folder.is_dir() or folder.name.startswith(('scale-','low-depth-main-r2-')):continue
        closure=root/'snapshots'/folder.name/'closure.json'
        if (not closure.exists() or closure.stat().st_mtime>=1793066985
                or json.loads(closure.read_text()).get('exit_code')!=0):continue
        for file in folder.glob('checkpoint*.pt'):
            if (file.is_symlink() or not file.is_file() or file.stat().st_size<500*1024**2
                    or file.with_suffix(file.suffix+'.exact.json').exists()
                    or str(file.relative_to(root)) in protected):continue
            candidates.append(dict(file=str(file.relative_to(root)),bytes=file.stat().st_size,
                                   closure=str(closure.relative_to(root)),closure_sha256=sha256_file(closure)))
    candidates.sort(key=lambda row:(-row['bytes'],row['file']));selected=[];total=0
    for row in candidates:
        if total+row['bytes']<=50*GIB:selected.append(row);total+=row['bytes']
    if not selected:raise ValueError('No remaining closed exact-preservation candidates')
    sample_dir=Path('/home/ubuntu/dario/local/nvme/.autonomous-compute-scale-private-20261027')
    if not sample_dir.is_dir() or sample_dir.stat().st_mode&0o077:raise ValueError('Existing private task output required')
    sample=sample_dir/(IDENTITY+'-restore.pt')
    if sample.exists():raise FileExistsError('Never overwrite a restoration sample')
    started=time.time();out.mkdir()
    plan=dict(identity=IDENTITY,registered_unix=started,source_sha256=sha256_file(Path(__file__)),
              tuning_report_sha256=tune_sha,scientific_scope_sha256=scope_sha,
              maximum_original_bytes=50*GIB,selected_original_bytes=total,original_inventory=selected,
              required_free_bytes=required_free_bytes,protected_raw_zip_dependency_parents=sorted(protected),
              maximum_parallel_streams=1,operational_seconds_limit=5400,
              assigned_chips=0,optimizer_updates=0,
              scope='One inactive-instance exact archive after complete tuning and maintenance closure,before any main admission. Only closed historical pre-scale checkpoint binaries,no source data or failed-state release. Stop once actual full main storage fits or50GiB original-byte allowance is exhausted. Both stopped/complete results and original binary hashes remain. No new scientific exposure or further automatic maintenance.')
    write_state(out/'plan.json',plan);records=[]
    result=dict(status='STOPPED_INACTIVE_EXACT_ARCHIVE',assigned_chips=0,optimizer_updates=0)
    def check(extra):
        if time.time()>started+5400:raise TimeoutError('Inactive archive operational watchdog')
        if (managed_running() or (root/'scale-body-exposure-main-v1').exists()
                or sha256_file(tuning_report)!=tune_sha or sha256_file(scope_file)!=scope_sha):
            raise RuntimeError('Inactive preservation allocation changed')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                       if s.startswith('MemAvailable:'))
        if available<512*GIB+256*1024**2:raise RuntimeError('Protected host reserve')
        if shutil.disk_usage(root).free<35*GIB+extra:raise RuntimeError('Protected output reserve plus full current compression peak')
    try:
        for i,row in enumerate(selected):
            if shutil.disk_usage(root).free>=required_free_bytes:break
            check(row['bytes']+64*1024**2);file=root/row['file']
            if sha256_file(root/row['closure'])!=row['closure_sha256']:raise AssertionError('Historical closure changed')
            recipe,record=create_archive(file)
            if i==0:
                if shutil.disk_usage(sample_dir).free<35*GIB+row['bytes']+64*1024**2:
                    raise RuntimeError('Full materialized restoration and reserve required')
                restore_archive(recipe,sample)
                if sha256_file(sample)!=record['original_sha256']:raise AssertionError('Materialized original differs')
                sample.unlink()
            if sha256_file(file)!=record['original_sha256']:raise AssertionError('Original changed before exact retention')
            file.unlink();records.append(dict(file=row['file'],**record))
            write_state(out/'progress.json',dict(completed=len(records),maximum_files=len(selected),records=records))
        if shutil.disk_usage(root).free<required_free_bytes:
            raise RuntimeError('Bounded exact preservation did not produce complete main allowance;no automatic extension')
        result['status']='COMPLETE_INACTIVE_EXACT_ARCHIVE_TARGET_REACHED'
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(records=records,wall_seconds=time.time()-started,files=len(records),
                      original_bytes=sum(r['original_bytes'] for r in records),
                      archive_bytes=sum(r['archive_bytes'] for r in records),
                      actual_root_free_after=shutil.disk_usage(root).free,original_sha256_restorable=True)
        write_state(out/'report.json',result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--required-free-bytes',type=int,required=True)
    p.add_argument('--tuning-report',type=Path,required=True);p.add_argument('--scope-file',type=Path,required=True)
    a=p.parse_args();archive(a.root,required_free_bytes=a.required_free_bytes,tuning_report=a.tuning_report,scope_file=a.scope_file)
