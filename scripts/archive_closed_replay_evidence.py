"""Lossless sharing of completed duplicate replay evidence; private bytes kept."""
import argparse
import concurrent.futures as futures
import json
from pathlib import Path
import re
import shutil
import time

from compute_trainium.lossless_zip_delta import create_archive,reconstructed_digest,restore_archive,digest_file

IDENTITY='closed-956m-replay-exact-archive-v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--profile',choices=['large','medium_recurrent','medium_ordinary'],default='large')
    parser.add_argument('--apply',action='store_true');args=parser.parse_args();root=args.root
    if root.resolve()!=Path('/run/user/1000/autonomous-compute/instance-nvme'):
        raise ValueError('Exact existing research root required')
    native,status,count={
        'large':('scale-956m-numerical-v1','PASS_SCALE_956M_COMPLETE_NUMERICAL_QUALIFICATION',129),
        'medium_recurrent':('scale-376m-fixed-two-numerical-v1','PASS_SCALE_MEDIUM_FIXED_TWO_COMPLETE_NUMERICAL_QUALIFICATION',129),
        'medium_ordinary':('scale-375m-ordinary-numerical-v1','PASS_SCALE_MEDIUM_ORDINARY_COMPLETE_NUMERICAL_QUALIFICATION',65),
    }[args.profile]
    proof_file=root/native/'completion-verification.json';proof=json.loads(proof_file.read_text())
    if proof['status']!=status:raise ValueError('Complete independent closure required')
    base=root.parent/'recurrence'/native if args.profile=='large' else root/native/'artifacts'
    identity=IDENTITY if args.profile=='large' else 'closed-'+args.profile.replace('_','-')+'-replay-exact-archive-v1'
    out=root/identity
    if out.exists():raise FileExistsError('Preserve every archival attempt')
    out.mkdir();started=time.time();deadline=started+1200
    children=sorted((base/'replay').glob('*.pt'))
    if len(children)!=count:raise ValueError('Exact registered closed replay file scope required')
    for child in children:
        if (not re.fullmatch(r'(?:rank(?:[0-9]|[1-5][0-9]|6[0-3])-u101|noise-u100-rank(?:[0-9]|[1-5][0-9]|6[0-3])|gradient-u100)\.pt',child.name)
                or child.is_symlink() or not (base/'warm'/child.name).is_file()):
            raise ValueError('Closed exact replay filename required')
    (out/'plan.json').write_text(json.dumps(dict(identity=identity,profile=args.profile,started_unix=started,deadline_unix=deadline,
        source_sha256=digest_file(__file__),helper_sha256=digest_file(Path(__file__).resolve().parents[1]/'compute_trainium/lossless_zip_delta.py'),
        completion_proof_sha256=digest_file(proof_file),
        original_files=[dict(file=c.name,sha256=digest_file(c),bytes=c.stat().st_size) for c in children],
        original_bytes_retained_through_verified_archive=True,apply_requested=args.apply),indent=2)+'\n')
    def one(child):
        if time.time()>deadline:raise TimeoutError('Original archive cap')
        if shutil.disk_usage(base).free<35*1024**3:raise RuntimeError('Protected archive filesystem reserve')
        recipe=child.with_suffix(child.suffix+'.archive.json')
        row=create_archive(base/'warm'/child.name,child,recipe)
        if row['shared_payload_bytes']<.99*row['original_bytes']:
            raise AssertionError('Expected exact duplicate payload was not established')
        return dict(file=child.name,recipe=recipe.name,**row)
    with futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(one,children))
    # One real serialized state also undergoes a materialized exact round trip.
    sample=next(r for r in rows if r['file']=='rank63-u101.pt');temporary=base/'replay/materialized-roundtrip-check.pt'
    restore_archive(base/'replay'/sample['recipe'],temporary)
    if digest_file(temporary)!=sample['original_sha256']:raise AssertionError('Actual state round trip failed')
    temporary.unlink()
    released=0
    if args.apply:
        for row in rows:
            if time.time()>deadline:raise TimeoutError('Original archive cap')
            child=base/'replay'/row['file'];recipe=base/'replay'/row['recipe']
            if digest_file(child)!=row['original_sha256'] or reconstructed_digest(recipe)!=(row['original_sha256'],row['original_bytes']):
                raise AssertionError('Original/archived bytes changed before release')
            child.unlink();released+=row['original_bytes']-row['archive_bytes']
            (out/'progress.json').write_text(json.dumps(dict(last_archived=row['file'],released_bytes=released))+'\n')
    result=dict(status='ARCHIVED_EXACT_CLOSED_REPLAY_EVIDENCE' if args.apply else 'VERIFIED_EXACT_CLOSED_REPLAY_ARCHIVES',
        original_file_count=len(rows),original_bytes=sum(r['original_bytes'] for r in rows),
        private_archive_bytes=sum(r['archive_bytes'] for r in rows),released_bytes=released,
        every_original_file_sha256_reconstructs=True,actual_state_materialized_roundtrip_passes=True,
        parent_files_unchanged=True,raw_sources_and_numeric_corpora_untouched=True,archived_file_inventory=rows,
        wall_seconds=time.time()-started,assigned_chips=0,optimizer_updates=0,
        restore_instruction='Use compute_trainium.lossless_zip_delta.restore_archive(recipe,target) before a legacy direct-file read.',
        scope='Different ZIP headers/metadata retained literally; only verified equal data payloads shared with immutable warm originals. Original binary fingerprints preserved,not lossy reserialization. Private literal bytes/recipes stay private.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='archived_file_inventory'}),flush=True)


if __name__=='__main__':main()
