"""Retire four regenerable caches, retaining results, labels and model history."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from compute_trainium.exact_gzip_archive import sha256_file, verified_stream

IDENTITY = 'closed-hidden-probe-feature-retirement-v1'


def main(root):
    start = time.time()
    old = root/'first-pass-hidden-value-probe-v1'
    plan = json.loads((old/'plan.json').read_text())
    result = json.loads((old/'report.json').read_text())
    if result['status'] != 'COMPLETE_TRAIN_SOURCE_HIDDEN_VALUE_INFORMATION_DIAGNOSTIC':
        raise AssertionError('Completed historical probe only')
    out = root/IDENTITY
    out.mkdir(mode=0o700)
    receipt = dict(status='PREPARING_CLOSED_FEATURE_RETIREMENT', started_unix=start,
        assigned_chips=0, model_forwards=0, optimizer_updates=0,
        original_plan_sha256=sha256_file(old/'plan.json'),
        original_result_sha256=sha256_file(old/'report.json'),
        scientific_states_modified=0, score_records_modified=0,
        scope='Delete only the four closed float16 hidden-feature caches at owner authorization. Preserve all published/private results, loss labels, original plans, source data and exact full checkpoints. Regeneration requires the retained historical code and data; no bit-exact regeneration is claimed by this maintenance. Removed cache hashes are retained; this is retirement, not exact archival.')
    (out/'plan.json').write_text(json.dumps(receipt,indent=2)+'\n')
    files = []
    try:
        if sha256_file(root/plan['data']['packed']/'manifest.json') != plan['data']['manifest_sha256']:
            raise AssertionError('Retained source manifest changed')
        for job in plan['jobs']:
            checkpoint = root/job['checkpoint']
            if checkpoint.exists():
                if sha256_file(checkpoint) != job['checkpoint_sha256']:
                    raise AssertionError('Retained full checkpoint changed')
            else:
                recipe = checkpoint.with_suffix(checkpoint.suffix+'.exact.json')
                if json.loads(recipe.read_text())['original_sha256'] != job['checkpoint_sha256']:
                    raise AssertionError('Archived original checkpoint identity differs')
                for _ in verified_stream(recipe):
                    if time.time()-start > 600:
                        raise TimeoutError('Ten-minute maintenance limit')
            folder = old/f"s{job['seed']}"
            proof = json.loads((folder/'report.json').read_text())
            if not proof['status'].startswith('COMPLETE_'):
                raise AssertionError('Completed per-seed result required')
            for name in ('train-features.npy','dev-features.npy'):
                file = folder/name
                if file.is_symlink() or not file.is_file():
                    raise AssertionError('Original regular feature cache required')
                stat = file.stat()
                digest = sha256_file(file)
                if digest != proof['private_trace_hashes'][name]:
                    raise AssertionError('Original cached feature bytes changed')
                files.append(dict(file=str(file.relative_to(root)),bytes=stat.st_size,
                    sha256=digest,identity=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]))
        if len(files)!=4 or sum(x['bytes'] for x in files)>4*1024**3:
            raise AssertionError('Four known caches below four GiB only')
        receipt['inventory']=files
        (out/'verified-inventory.json').write_text(json.dumps(receipt,indent=2)+'\n')
        before=shutil.disk_usage(root).free
        for row in files:
            if time.time()-start>600:
                raise TimeoutError('Ten-minute maintenance limit')
            file=root/row['file'];s=file.stat()
            if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]!=row['identity']:
                raise AssertionError('Closed cache changed during verification')
            file.unlink()
        receipt.update(status='COMPLETE_CLOSED_FEATURE_RETIREMENT',removed_files=4,
            removed_bytes=sum(x['bytes'] for x in files),free_bytes_before=before,
            free_bytes_after=shutil.disk_usage(root).free)
    except BaseException as error:
        receipt.update(status='STOPPED_CLOSED_FEATURE_RETIREMENT',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        receipt['wall_seconds']=time.time()-start
        (out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:v for k,v in receipt.items() if k!='inventory'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
