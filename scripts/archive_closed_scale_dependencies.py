"""Bounded exact compression of six closed, explicitly named evidence sets."""
import argparse
import json
from pathlib import Path
import shutil
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;started=time.time()
    from compute_trainium.exact_gzip_archive import create_archive,restore_archive,sha256_file
    targets=[]
    for name in ['scale-376m-fixed-two-numerical-v1','scale-375m-ordinary-numerical-v1']:
        proof=root/name/'completion-verification.json'
        if not json.loads(proof.read_text())['status'].startswith('PASS_SCALE_MEDIUM_'):raise ValueError('Closed original medium qualification required')
        targets.append((root/name/'artifacts/warm',proof))
    for identity in ['head-free-fixed2-numerical-v2-recovery-half-a-warm','head-free-fixed2-numerical-v2-recovery-half-b-warm',
                     'local-batch-qualification-v2-one-b2-warm','local-batch-qualification-v2-one-b4-warm']:
        closure=root/'snapshots'/identity/'closure.json'
        if not closure.is_file():raise ValueError('Owned original closure must remain present')
        json.loads(closure.read_text())['exit_code']
        targets.append((root/'runs'/identity,closure))
    # The new ordinary admission reads original size receipts once. Leave its
    # parent files available until that immutable registration has been made.
    if not (root/'scale-65536-ordinary-numerical-v1/plan.json').exists():
        raise ValueError('Register the unchanged ordinary size inventory before closing old direct-file dependencies')
    out=root/'closed-scale-dependencies-exact-archive-v1'
    if out.exists():raise FileExistsError('Preserve every maintenance attempt')
    files=[(f,proof) for folder,proof in targets for f in sorted(folder.glob('*.pt')) if not f.is_symlink()]
    total=sum(f.stat().st_size for f,_ in files)
    if total>80*1024**3 or not files:raise ValueError('Only the declared closed80GiB scope may be compressed')
    out.mkdir();deadline=started+1200
    inventory=[dict(file=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha256_file(f),closure_sha256=sha256_file(proof)) for f,proof in files]
    (out/'plan.json').write_text(json.dumps(dict(purpose='exact_closed_binary_storage_only',started_unix=started,
        absolute_deadline_unix=deadline,maximum_original_bytes=80*1024**3,original_inventory=inventory,
        original_outcomes_including_failures_unchanged=True,no_source_corpus_or_unique_binary_discarded=True,
        assigned_chips=0,optimizer_updates=0),indent=2)+'\n')
    rows=[];sample_done=False
    for file,_ in files:
        if time.time()>deadline:raise TimeoutError('Original20-minute maintenance cap')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<512*1024**3+2*file.stat().st_size+64*1024**2:raise RuntimeError('Protected host reserve plus complete temporary archive/sample workspace')
        if shutil.disk_usage(file).free<35*1024**3+2*file.stat().st_size+64*1024**2:raise RuntimeError('Complete temporary archive/restore allowance plus protected disk reserve')
        recipe,row=create_archive(file)
        if not sample_done:
            sample=file.parent/'exact-binary-restoration-check.pt';restore_archive(recipe,sample)
            if sha256_file(sample)!=row['original_sha256']:raise AssertionError('Actual binary restoration differs')
            sample.unlink();sample_done=True
        if sha256_file(file)!=row['original_sha256']:raise AssertionError('Original changed before exact compressed retention')
        rows.append(dict(file=str(file.relative_to(root)),**row));(out/'progress.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n');file.unlink()
    report=dict(status='ARCHIVED_CLOSED_DEPENDENCIES_WITH_EXACT_ORIGINAL_BYTES',files=len(rows),original_bytes=total,
        archive_bytes=sum(r['archive_bytes'] for r in rows),released_bytes=total-sum(r['archive_bytes'] for r in rows),
        every_original_binary_sha256_restorable=True,materialized_sample_exact=True,original_outcomes_and_receipts_unchanged=True,
        root_free_after=shutil.disk_usage(root).free,assigned_chips=0,optimizer_updates=0,wall_seconds=time.time()-started,
        restore_instruction='Restore original warm files using exact_gzip_archive.restore_archive before legacy direct reads or replay ZIP-archive restoration.',records=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
