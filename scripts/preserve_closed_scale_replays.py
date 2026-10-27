"""Exact binary replay sharing for two closed scale histories,including rejects."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time
import zipfile

from compute_trainium.execution_isolation import managed_running
from compute_trainium.lossless_zip_delta import create_archive,digest_file,restore_archive
from compute_trainium.queue import write_state

IDENTITY='closed-legacy-zip-scale-replays-v1'
PUBLIC=Path('results/workspace_storage_preservation_20261027')
CASES=[('scale-boundary-rate-completion-v2','fixed-two-lr2'),
       ('scale-boundary-last-original-rates-v1','fixed-two-lr3')]
GIB=1024**3


def main(root):
    started=time.time();folder=root/IDENTITY
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every exact-sharing attempt')
    with (root/'native-half-reservations.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if managed_running():raise RuntimeError('Closed maintenance requires every Native owner terminal')
        selected=[];closures={}
        for identity,key in CASES:
            plan=json.loads((root/identity/'plan.json').read_text())
            if not (root/identity/'report.json').exists():raise ValueError('Both historical controllers must be terminal')
            for phase in plan['case_native_phases'][key]:
                file=root/'snapshots'/phase/'closure.json'
                if json.loads(file.read_text())['exit_code']!=0:raise ValueError('Both Native captures must have closed successfully')
                closures[str(file.relative_to(root))]=digest_file(file)
            base=root/identity/'artifacts'/key
            for rank in range(64):
                parent=base/'warm'/f'rank{rank}-u300.pt';child=base/'replay'/parent.name
                for file in [parent,child]:
                    if file.is_symlink() or not file.is_file():raise ValueError('Complete original rank pair required')
                    note=json.loads(file.with_suffix('.json').read_text())
                    if digest_file(file)!=note['sha256']:raise AssertionError('Original retained shard differs from its receipt')
                with zipfile.ZipFile(parent) as pzip,zipfile.ZipFile(child) as czip:
                    peers={i.filename.split('/',1)[1]:i for i in pzip.infolist()}
                    shared=sum(i.compress_size for i in czip.infolist() if i.filename.split('/',1)[1] in peers
                        and (peers[i.filename.split('/',1)[1]].CRC,peers[i.filename.split('/',1)[1]].compress_size)==(i.CRC,i.compress_size))
                if shared<.9*child.stat().st_size:raise ValueError('No high-confidence duplicate payload candidate')
                selected.append(dict(parent=str(parent.relative_to(root)),child=str(child.relative_to(root)),bytes=child.stat().st_size))
        if sum(v['bytes'] for v in selected)>8*GIB:raise ValueError('Original-child allowance exceeds8GiB')
        folder.mkdir();PUBLIC.mkdir();plan=dict(identity=IDENTITY,registered_unix=time.time(),
            source_sha256={f:digest_file(f) for f in ['scripts/preserve_closed_scale_replays.py','compute_trainium/lossless_zip_delta.py']},
            selected=selected,maximum_original_child_bytes=8*GIB,maximum_seconds=900,closure_sha256=closures,
            raw_dependency_parents_must_remain=[r['parent'] for r in selected],assigned_chips=0,optimizer_updates=0,
            scope='Only two closed Native replay captures. Includes numerical rejects without discarding their values or history. Exact payload hashes and full original binary reconstruction required. Raw warm parents protected. No source,corpus,model value,optimizer value or result changes.')
        write_state(folder/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
        result=dict(status='STOPPED_CLOSED_SCALE_REPLAY_PRESERVATION',records=[],assigned_chips=0,optimizer_updates=0)
        sample_dir=Path('/home/ubuntu/dario/local/nvme/.autonomous-compute-scale-private-20261027')
        try:
            for i,row in enumerate(selected):
                if time.time()-started>900 or managed_running():raise RuntimeError('Bounded maintenance or exclusive allocation changed')
                available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
                if available<512*GIB or shutil.disk_usage(root).free<35*GIB+row['bytes']+64*1024**2:
                    raise RuntimeError('Full current archive peak plus protected reserves required')
                parent,child=root/row['parent'],root/row['child'];recipe=child.with_suffix('.pt.zipdelta.json')
                record=create_archive(parent,child,recipe)
                if record['shared_payload_bytes']<.9*record['original_bytes']:raise AssertionError('Exact shared payload falls below its screen')
                if i==0:
                    if sample_dir.stat().st_mode&0o077 or shutil.disk_usage(sample_dir).free<35*GIB+row['bytes']+64*1024**2:
                        raise RuntimeError('Private full restoration peak required')
                    sample=sample_dir/(IDENTITY+'-restore.pt');restore_archive(recipe,sample)
                    if digest_file(sample)!=record['original_sha256']:raise AssertionError('Materialized original differs')
                    sample.unlink()
                if digest_file(child)!=record['original_sha256']:raise AssertionError('Child changed before exact retention')
                child.unlink();result['records'].append(dict(**row,recipe=str(recipe.relative_to(root)),**record))
                write_state(folder/'progress.json',dict(completed=len(result['records']),total=len(selected)))
            for file,sha in closures.items():
                if digest_file(root/file)!=sha:raise AssertionError('Historical closure changed')
            result['status']='COMPLETE_CLOSED_SCALE_REPLAY_PRESERVATION'
        except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
        finally:
            result.update(files=len(result['records']),wall_seconds=time.time()-started,
                recovered_bytes=sum(r['original_bytes']-r['archive_bytes'] for r in result['records']),
                raw_dependency_parents_retained=True,original_binary_restoration_verified=True,
                actual_root_free_after=shutil.disk_usage(root).free)
            write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',{k:v for k,v in result.items() if k!='records'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
