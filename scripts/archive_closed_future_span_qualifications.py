"""Bounded four-way streaming archive of four explicitly closed qualifications."""
import concurrent.futures
import json
from pathlib import Path
import shutil
import threading
import time


def main(root):
    from compute_trainium.exact_gzip_archive import create_archive,restore_archive,sha256_file
    names=['future-span-math-qualification-v2-one-ntp-warm','future-span-math-qualification-v2-one-future-warm',
           'future-span-math-qualification-v2-recurrent-ntp-warm','future-span-math-qualification-v2-recurrent-future-warm']
    started=time.time();deadline=started+1800;out=root/'closed-future-span-qualifications-exact-archive-v1'
    if out.exists():raise FileExistsError('Preserve every maintenance attempt')
    files=[];closures={}
    for name in names:
        proof=root/'snapshots'/name/'closure.json'
        if json.loads(proof.read_text())['exit_code']!=0:raise ValueError('Original complete owned closure required')
        closures[name]=sha256_file(proof)
        files.extend(f for f in sorted((root/'runs'/name).glob('*.pt')) if not f.is_symlink())
    if not files or sum(f.stat().st_size for f in files)>40*1024**3:raise ValueError('Only the declared40GiB closed scope')
    out.mkdir();inventory=[dict(file=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha256_file(f)) for f in files]
    plan=dict(started_unix=started,absolute_deadline_unix=deadline,original_inventory=inventory,closures=closures,
        assigned_chips=0,optimizer_updates=0,maximum_parallel_streams=4,maximum_original_bytes=40*1024**3,
        original_results_unchanged=True,all_original_binary_bytes_retained=True)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    lock=threading.Lock();reserved=0;rows=[]
    def archive(item):
        nonlocal reserved
        file,index=item;needed=2*file.stat().st_size+64*1024**2
        with lock:
            if time.time()>deadline:raise TimeoutError('Original30-minute maintenance cap')
            available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            if available<512*1024**3+256*1024**2:raise RuntimeError('Protected host reserve plus streaming workspace')
            if shutil.disk_usage(root).free<35*1024**3+reserved+needed:raise RuntimeError('All admitted temporary binary copies plus disk reserve must fit')
            reserved+=needed
        try:
            recipe,row=create_archive(file)
            if index==0:
                sample=file.parent/'exact-archive-materialized-check.pt';restore_archive(recipe,sample)
                if sha256_file(sample)!=row['original_sha256']:raise AssertionError('Actual restored binary differs')
                sample.unlink()
            if sha256_file(file)!=row['original_sha256']:raise AssertionError('Closed original changed before exact retention')
            file.unlink()
            with lock:
                rows.append(dict(file=str(file.relative_to(root)),**row))
                (out/'progress.json').write_text(json.dumps(dict(completed_files=len(rows),total_files=len(files),records=rows),indent=2)+'\n')
        finally:
            with lock:reserved-=needed
    result=dict(status='STOPPED_CLOSED_QUALIFICATION_ARCHIVE',original_outcomes_unchanged=True,assigned_chips=0,optimizer_updates=0)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for _ in pool.map(archive,[(f,i) for i,f in enumerate(files)]):pass
        result['status']='ARCHIVED_CLOSED_QUALIFICATIONS_EXACT_BINARIES'
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(files=len(rows),original_bytes=sum(r['original_bytes'] for r in rows),
            archive_bytes=sum(r['archive_bytes'] for r in rows),wall_seconds=time.time()-started,
            all_completed_original_sha256_restorable=True,records=rows)
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
