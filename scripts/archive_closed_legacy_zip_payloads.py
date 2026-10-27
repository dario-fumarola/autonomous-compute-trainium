"""Bounded exact ZIP sharing for closed replay copies, never source data."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from compute_trainium.lossless_zip_delta import create_archive, digest_file, restore_archive
from compute_trainium.queue import write_state

IDENTITY = 'closed-legacy-zip-payload-preservation-v1'
GIB = 1024**3


def main(root):
    folder = root / IDENTITY
    if folder.exists():raise FileExistsError('Preserve every archival attempt')
    stage = root / 'scale-boundary-rate-completion-v2'
    current = json.loads((stage / 'plan.json').read_text())
    pid = int((stage / 'controller.pid').read_text())
    gzip_plan = json.loads((root / 'closed-legacy-checkpoint-exact-archive-v1/plan.json').read_text())
    excluded = {r['file'] for r in gzip_plan['original_inventory']}
    screen_file = root / 'closed-legacy-zip-metadata-screen-v1/report.json'
    screen = json.loads(screen_file.read_text())
    selected = []; parents = set(); children = set(); total = 0
    for row in screen['best']:
        if (row['parent'] in excluded or row['child'] in excluded
                or row['parent'] in children or row['child'] in children or row['child'] in parents
                or row['crc_candidate_shared_bytes'] < .9 * row['child_bytes']
                or total + row['child_bytes'] > 24 * GIB):continue
        closures = {}
        for name in [row['parent'], row['child']]:
            file = root / name
            closure = root / 'snapshots' / file.parent.name / 'closure.json'
            if (not file.is_file() or file.is_symlink() or closure.stat().st_mtime >= 1793066985
                    or json.loads(closure.read_text())['exit_code'] != 0):
                raise ValueError('Only unchanged terminal historical files with exact successful closure')
            closures[str(closure.relative_to(root))] = digest_file(closure)
        selected.append(dict(row, closure_sha256=closures))
        parents.add(row['parent']); children.add(row['child']); total += row['child_bytes']
    if not selected:raise ValueError('No useful non-overlapping exact-sharing candidates')
    sample_dir = Path('/home/ubuntu/dario/local/nvme/.autonomous-compute-scale-private-20261027')
    if not sample_dir.is_dir() or sample_dir.stat().st_mode & 0o077:
        raise ValueError('Existing private task output required')
    sample = sample_dir / (IDENTITY + '-restore.pt')
    if sample.exists():raise FileExistsError('Never overwrite a restore sample')
    started = time.time(); folder.mkdir()
    plan = dict(identity=IDENTITY, registered_unix=started, source_sha256=digest_file(Path(__file__)),
                metadata_screen_sha256=digest_file(screen_file), selected=selected,
                maximum_original_child_bytes=24*GIB, selected_original_child_bytes=total,
                current_stage_plan_sha256=digest_file(stage/'plan.json'), current_stage_pid=pid,
                protected_filesystem_bytes=35*GIB, protected_host_bytes=512*GIB,
                active_stage_artifact_peak_bytes=18*GIB,
                concurrent_gzip_peak_bytes=max(r['bytes'] for r in gzip_plan['original_inventory']),
                raw_dependency_parents_must_remain=sorted(parents), maximum_parallel_zip_streams=1,
                optimizer_updates=0, assigned_chips=0,
                scope='Only listed closed historical checkpoint children, excluding every concurrent gzip input. Exact ZIP payload SHA matching and whole original binary reconstruction before releasing raw duplicate; materialize first restoration separately. All raw parents stay protected. No corpus, model values, optimizer values, scientific result or failure changes. Stop before a new file when declared active stage terminates; no unknown successor overlap.')
    write_state(folder/'plan.json',plan); records=[]
    result=dict(status='STOPPED_CLOSED_LEGACY_ZIP_PRESERVATION',assigned_chips=0,optimizer_updates=0)
    def reserve(extra):
        while True:
            if (time.time()>started+3600 or (stage/'report.json').exists()
                    or subprocess.run(['kill','-0',str(pid)],capture_output=True).returncode):
                raise RuntimeError('Declared stage closed or bounded maintenance watchdog ended')
            if digest_file(stage/'plan.json')!=plan['current_stage_plan_sha256']:
                raise AssertionError('Declared active allocation changed')
            committed=sum(f.stat().st_size for f in (stage/'artifacts').rglob('*.pt')
                          if f.is_file() and not f.is_symlink())
            peak=max(0,18*GIB-committed)
            available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                           if s.startswith('MemAvailable:'))
            if (available>=512*GIB+256*1024**2
                    and shutil.disk_usage(root).free>=35*GIB+peak+plan['concurrent_gzip_peak_bytes']+extra):return
            time.sleep(5)
    try:
        for i,row in enumerate(selected):
            reserve(row['child_bytes']+64*1024**2)
            for name,sha in row['closure_sha256'].items():
                if digest_file(root/name)!=sha:raise AssertionError('Historical closure changed')
            parent,child=root/row['parent'],root/row['child']
            recipe=child.with_suffix(child.suffix+'.zipdelta.json')
            record=create_archive(parent,child,recipe)
            if record['shared_payload_bytes']<.9*record['original_bytes']:
                raise AssertionError('Candidate failed exact shared-payload expectation; preserve all files')
            if i==0:
                if shutil.disk_usage(sample_dir).free<35*GIB+row['child_bytes']+64*1024**2:
                    raise RuntimeError('Complete materialized restoration plus reserve required')
                restore_archive(recipe,sample)
                if digest_file(sample)!=record['original_sha256']:raise AssertionError('Materialized bytes differ')
                sample.unlink()
            if digest_file(child)!=record['original_sha256']:raise AssertionError('Original child changed')
            child.unlink()
            records.append(dict(parent=row['parent'],child=row['child'],recipe=str(recipe.relative_to(root)),**record))
            write_state(folder/'progress.json',dict(completed=len(records),total=len(selected),records=records))
        result['status']='COMPLETE_CLOSED_LEGACY_ZIP_PRESERVATION'
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(records=records,files=len(records),wall_seconds=time.time()-started,
                      original_bytes=sum(r['original_bytes'] for r in records),
                      archive_bytes=sum(r['archive_bytes'] for r in records),
                      dependency_parents_retained=True,exact_original_restoration_verified=True)
        write_state(folder/'report.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
