"""Bounded exact archival of terminal historical checkpoints,never source data."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from compute_trainium.exact_gzip_archive import create_archive,restore_archive,sha256_file

IDENTITY='closed-legacy-checkpoint-exact-archive-v1'
GIB=1024**3


def main(root,*,identity=IDENTITY,stage_name='scale-boundary-rate-completion-v2',maximum_gib=120):
    if not identity.startswith('closed-legacy-checkpoint-exact-archive-v'):
        raise ValueError('Explicit legacy exact-archive identity required')
    if stage_name not in ['scale-boundary-rate-completion-v2','scale-boundary-last-original-rates-v1','scale-boundary-lower-rates-v1']:
        raise ValueError('Explicit qualified rate-stage allocation required')
    if not isinstance(maximum_gib,int) or not 1<=maximum_gib<=200:
        raise ValueError('Explicit bounded original-byte scope at most200GiB required')
    out=root/identity
    if out.exists():raise FileExistsError('Preserve every archival attempt')
    stage=root/stage_name;pid=int((stage/'controller.pid').read_text());candidates=[]
    protected=set()
    for file in root.glob('closed-legacy-zip-*/plan.json'):
        protected.update(json.loads(file.read_text()).get('raw_dependency_parents_must_remain',[]))
    for folder in (root/'runs').iterdir():
        if not folder.is_dir() or folder.name.startswith(('scale-','low-depth-main-r2-')):continue
        closure=root/'snapshots'/folder.name/'closure.json'
        if not closure.exists() or closure.stat().st_mtime>=1793066985:continue
        if json.loads(closure.read_text()).get('exit_code')!=0:continue
        for file in folder.glob('checkpoint*.pt'):
            if file.is_symlink() or not file.is_file() or file.stat().st_size<500*1024**2:continue
            if file.with_suffix(file.suffix+'.exact.json').exists():continue
            if str(file.relative_to(root)) in protected:continue
            candidates.append(dict(file=str(file.relative_to(root)),bytes=file.stat().st_size,closure=str(closure.relative_to(root)),closure_sha256=sha256_file(closure)))
    candidates.sort(key=lambda r:(-r['bytes'],r['file']));selected=[];total=0
    for row in candidates:
        if total+row['bytes']<=maximum_gib*GIB:selected.append(row);total+=row['bytes']
    if not selected:raise ValueError('No complete closed original scope')
    # Materialize the first restoration on a separate,already task-owned
    # private filesystem;the compressed stream also verifies every byte.
    selected.sort(key=lambda r:(r['bytes'],r['file']))
    sample_parent=Path('/home/ubuntu/dario/local/nvme/.autonomous-compute-scale-private-20261027')
    if not sample_parent.is_dir() or sample_parent.stat().st_mode&0o077:raise ValueError('Existing private700 task directory required')
    sample=sample_parent/(identity+'-materialized-restore.pt')
    if sample.exists():raise FileExistsError('Never overwrite any existing task evidence')
    out.mkdir();plan=dict(identity=identity,registered_unix=time.time(),source_sha256=sha256_file(Path(__file__)),
        maximum_original_bytes=maximum_gib*GIB,selected_original_bytes=total,original_inventory=selected,
        current_stage=stage_name,current_stage_plan_sha256=sha256_file(stage/'plan.json'),
        protected_raw_zip_dependency_parents=sorted(protected),
        native_artifact_allowance_bytes=18*GIB,protected_filesystem_bytes=35*GIB,protected_host_bytes=512*GIB,
        maximum_parallel_streams=1,assigned_chips=0,optimizer_updates=0,
        scope='Only terminal pre-October27,02:09:45 native checkpoint binaries with successful exact owned closures. Original gzip stream/hash and a materialized first sample verified before releasing raw duplicate. No source files,parameters,optimizer values,scientific results or failures change. Active-stage artifact credit is only for reserving its remaining18GiB peak,never for main admission. Stop if this stage is terminal;no overlapping unknown successor allowed.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');records=[];wait_seconds=0.;started=time.time()
    def active():
        if (stage/'report.json').exists() or subprocess.run(['kill','-0',str(pid)],capture_output=True).returncode:raise RuntimeError('Declared active stage is terminal;stop maintenance before any unknown successor')
        if sha256_file(stage/'plan.json')!=plan['current_stage_plan_sha256']:raise AssertionError('Declared active allocation changed')
    def reserve(extra):
        nonlocal wait_seconds
        while True:
            active()
            committed=sum(f.stat().st_size for f in (stage/'artifacts').rglob('*.pt') if f.is_file() and not f.is_symlink())
            remaining=max(0,18*GIB-committed);available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            if available>=512*GIB+256*1024**2 and shutil.disk_usage(root).free>=35*GIB+remaining+extra:return
            time.sleep(5);wait_seconds+=5
    result=dict(status='STOPPED_CLOSED_LEGACY_EXACT_ARCHIVE',original_results_unchanged=True,assigned_chips=0,optimizer_updates=0)
    try:
        for index,row in enumerate(selected):
            file=root/row['file'];reserve(row['bytes']+64*1024**2)
            if sha256_file(root/row['closure'])!=row['closure_sha256']:raise AssertionError('Original exact terminal closure changed')
            if index==0 and shutil.disk_usage(sample_parent).free<35*GIB+row['bytes']+64*1024**2:raise RuntimeError('Full separate restoration sample and35GiB reserve required')
            recipe,record=create_archive(file)
            if index==0:
                restore_archive(recipe,sample)
                if sha256_file(sample)!=record['original_sha256']:raise AssertionError('Materialized original bytes differ')
                sample.unlink()
            if sha256_file(file)!=record['original_sha256']:raise AssertionError('Original changed before exact retention')
            file.unlink();records.append(dict(file=row['file'],**record))
            (out/'progress.json').write_text(json.dumps(dict(completed_files=len(records),total_files=len(selected),records=records),indent=2)+'\n')
        result['status']='COMPLETE_CLOSED_LEGACY_EXACT_ARCHIVE'
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(files=len(records),original_bytes=sum(x['original_bytes'] for x in records),archive_bytes=sum(x['archive_bytes'] for x in records),
            all_completed_original_sha256_restorable=True,records=records,wait_seconds=wait_seconds,wall_seconds=time.time()-started)
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--identity',default=IDENTITY);p.add_argument('--stage',default='scale-boundary-rate-completion-v2')
    p.add_argument('--maximum-gib',type=int,default=120)
    a=p.parse_args();main(a.root,identity=a.identity,stage_name=a.stage,maximum_gib=a.maximum_gib)
