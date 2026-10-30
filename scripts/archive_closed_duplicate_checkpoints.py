"""Reclaim only verified duplicate closed tensors, retaining every original byte."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from compute_trainium.lossless_zip_delta import create_archive,digest_file,reconstructed_digest,restore_archive
from compute_trainium.queue import write_state

ID='closed-duplicate-checkpoints-20261030-v1'
PUBLIC=Path('results/closed_duplicate_storage_20261030')
GIB=1024**3


def main(root):
    start=time.time();out=root/ID;screen=root/'closed-legacy-duplicate-screen-20261030-v1'
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every maintenance attempt')
    active=root/'latent-exposure-work-recovery-v1';active_plan=json.loads((active/'plan.json').read_text())
    if (active/'report.json').exists():raise RuntimeError('Known active stage already closed')
    # Discover every dependency by schema, not an assumed archive filename.
    protected=set();recipes=0
    for folder,dirs,files in os.walk(root):
        if Path(folder).relative_to(root).parts[:1]==('snapshots',):dirs[:]=[];continue
        for name in files:
            if not name.endswith('.json'):continue
            path=Path(folder)/name
            if path.is_symlink():continue
            with path.open('rb') as stream:prefix=stream.read(512)
            if b'"schema": "exact_zip_payload_archive_v1"' not in prefix:continue
            record=json.loads(path.read_text());protected.add((path.parent/record['parent_relative']).resolve());recipes+=1
    groups=json.loads((screen/'PRIVATE-groups.json').read_text());selected=[];parents=set();total=0
    for group in groups:
        members=sorted(group,key=lambda r:(str((root/r['file']).resolve()) not in {str(p) for p in protected},r['file']))
        parent=members[0]
        for child in members[1:]:
            if (root/child['file']).resolve() in protected:continue
            if total+child['bytes']>12*GIB:continue
            selected.append(dict(parent=parent,child=child));parents.add(parent['file']);total+=child['bytes']
    selected.sort(key=lambda r:(r['child']['bytes'],r['child']['file']))
    if not selected or parents&{r['child']['file'] for r in selected}:raise AssertionError('Nonempty acyclic exact sharing required')
    for pair in selected:
        for row in pair.values():
            f=root/row['file'];c=root/row['closure']
            if not f.is_file() or f.is_symlink() or f.stat().st_size!=row['bytes'] or json.loads(c.read_text()).get('exit_code')!=0:
                raise AssertionError('Original successful closure and unchanged regular file required')
            row['closure_sha256']=digest_file(c)
    plan=dict(identity=ID,registered_unix=start,absolute_deadline_unix=start+1200,selected=selected,
        source_sha256={f:digest_file(f) for f in ('scripts/archive_closed_duplicate_checkpoints.py','compute_trainium/lossless_zip_delta.py','docs/CLOSED_DUPLICATE_ARCHIVE_20261030.md')},
        metadata_sha256=digest_file(screen/'PRIVATE-groups.json'),active_plan_sha256=digest_file(active/'plan.json'),
        raw_dependency_parents_must_remain=sorted(parents),existing_archive_dependencies=len(protected),
        maximum_original_bytes=12*GIB,active_remaining_artifact_allowance_bytes=int(1.5*GIB),
        scope=Path('docs/CLOSED_DUPLICATE_ARCHIVE_20261030.md').read_text())
    out.mkdir(mode=0o700);PUBLIC.mkdir(parents=True);write_state(out/'plan.json',plan)
    result=dict(status='STOPPED_CLOSED_DUPLICATE_ARCHIVE');records=[]
    def guard(extra=0):
        available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
        if (time.time()>plan['absolute_deadline_unix'] or available<512*GIB+64*1024**2
            or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*GIB
            or shutil.disk_usage(root).free<35*GIB+plan['active_remaining_artifact_allowance_bytes']+extra
            or min(os.statvfs(root).f_favail,os.statvfs('/').f_favail)<10100):
            raise RuntimeError('Protected archive resource bounds')
    try:
        for i,pair in enumerate(selected):
            if (active/'report.json').exists():raise RuntimeError('Known stage closed; do not overlap a successor')
            if digest_file(active/'plan.json')!=plan['active_plan_sha256']:raise AssertionError('Active allocation changed')
            parent,child=root/pair['parent']['file'],root/pair['child']['file']
            guard((2 if i==0 else 1)*child.stat().st_size+64*1024**2)
            for row in pair.values():
                if digest_file(root/row['closure'])!=row['closure_sha256']:raise AssertionError('Successful closure changed')
            recipe=child.with_suffix(child.suffix+'.20261030.zipdelta.json')
            row=create_archive(parent,child,recipe)
            if row['shared_payload_bytes']<.99*row['original_bytes']:
                raise AssertionError('Exact sharing below99%; retain original')
            if i==0:
                sample=out/'PRIVATE-materialized-roundtrip.pt';restore_archive(recipe,sample)
                if digest_file(sample)!=row['original_sha256']:raise AssertionError('Materialized exact-byte restoration failed')
                sample.unlink()
            if (digest_file(child)!=row['original_sha256'] or reconstructed_digest(recipe)!=(row['original_sha256'],row['original_bytes'])):
                raise AssertionError('Original/reconstructed binary changed before release')
            child.unlink();records.append(dict(parent=str(parent.relative_to(root)),child=str(child.relative_to(root)),recipe=str(recipe.relative_to(root)),**row))
            write_state(out/'progress.json',dict(completed_files=len(records),total_files=len(selected),
                recovered_bytes=sum(r['original_bytes']-r['archive_bytes'] for r in records)))
        if any(digest_file(f)!=h for f,h in active_plan['source_sha256'].items()):raise AssertionError('Active scientific source changed')
        result['status']='COMPLETE_EXACT_CLOSED_DUPLICATE_ARCHIVE'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(records=records,files=len(records),original_bytes=sum(r['original_bytes'] for r in records),
            archive_bytes=sum(r['archive_bytes'] for r in records),recovered_bytes=sum(r['original_bytes']-r['archive_bytes'] for r in records),
            wall_seconds=time.time()-start,assigned_chips=0,optimizer_updates=0,source_bytes_removed=0,
            every_completed_original_binary_reconstructs=True,raw_dependency_parents_retained=True,
            free_bytes_after=shutil.disk_usage(root).free,plan_sha256=digest_file(out/'plan.json'))
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',{k:v for k,v in result.items() if k!='records'})
        print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
