"""Verify/release only closed, immutable, publicly retrievable download caches."""
import argparse
import concurrent.futures as futures
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
import urllib.error
import urllib.parse
import urllib.request

from compute_trainium.training import file_sha256

IDENTITY='closed-public-web-cache-maintenance-v1'


def validate_candidate(cache,record):
    path=Path(record['path']);receipt=record['receipt']
    if (record['domain']!='web' or path.parent!=cache or not re.fullmatch(r'web-[0-9]+\.parquet',path.name)
            or path.is_symlink() or path.resolve().parent!=cache.resolve()):
        raise ValueError('Only declared plain web-Parquet download cache files')
    if (receipt['repository']!='HuggingFaceFW/fineweb-edu'
            or not re.fullmatch('[0-9a-f]{40}',receipt['revision'])
            or not re.fullmatch('[0-9a-f]{64}',receipt['sha256'])
            or '..' in Path(receipt['file']).parts or receipt['file'].startswith('/')):
        raise ValueError('Pinned public immutable dataset receipt required')
    info=path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1:
        raise ValueError('Only singly linked current-owner regular cache files')
    return path,receipt,info


def remote_hash(receipt):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    url='https://huggingface.co/datasets/'+receipt['repository']+'/resolve/'+receipt['revision']+'/'+urllib.parse.quote(receipt['file'],safe='/')
    opener=urllib.request.build_opener(NoRedirect)
    try:response=opener.open(urllib.request.Request(url,method='HEAD'),timeout=20)
    except urllib.error.HTTPError as error:response=error
    with response:
        etag=(response.headers.get('x-linked-etag') or response.headers.get('etag') or '').strip('"')
        if response.code not in (200,302,307) or etag!=receipt['sha256']:
            raise RuntimeError('Unauthenticated immutable remote content hash could not be verified')
        return response.code


def verify(root):
    out=root/IDENTITY
    if out.exists():raise FileExistsError('Preserve each cache verification')
    source=root/'natural-append-17000m-v3';cache=source/'raw'
    failure=json.loads((source/'failure.json').read_text())
    capacity=root/'natural-append-capacity-v4'
    retention=json.loads((capacity/'manifest.json').read_text()).get('scope','').lower()
    if 'raw sources' in retention and 'indispensable' in retention:
        raise ValueError('Source manifest retains raw sources as indispensable dependencies; do not release them')
    if (failure['status']!='STOPPED_CHAINED_DATA_PREPARATION'
            or json.loads((capacity/'independent-audit.json').read_text())['status']!='PASS_INDEPENDENT_CHAINED_EXTENSION_AUDIT'
            or json.loads((capacity/'subset-audit.json').read_text())['status']!='PASS_INDEPENDENT_CAPACITY_SUBSET_PROOF'):
        raise ValueError('Closed failed download and completed derived numeric audits required')
    from compute_trainium.execution_isolation import managed_running
    if managed_running():raise RuntimeError('Do not compete with an active Native owner')
    started=time.time();records=json.loads((source/'raw-source-map.json').read_text())
    selected=[r for r in records if Path(r['path']).parent==cache and re.fullmatch(r'web-[0-9]+\.parquet',Path(r['path']).name)]
    if len(selected)!=28:raise ValueError('Exact observed 28-file closed cache scope required')
    preserved=[source/'raw-source-map.json',source/'failure.json',source/'plan.json',capacity/'manifest.json',
        capacity/'independent-audit.json',capacity/'subset-audit.json']
    binding={str(p.relative_to(root)):file_sha256(p) for p in preserved}
    def one(record):
        path,receipt,info=validate_candidate(cache,record)
        if time.time()-started>1200:raise TimeoutError('Original 20-minute cache verification cap')
        if file_sha256(path)!=receipt['sha256']:raise AssertionError('Original local cache fingerprint differs')
        status=remote_hash(receipt)
        return dict(file=path.name,receipt=receipt,bytes=info.st_size,inode=info.st_ino,device=info.st_dev,
            remote_status=status,unauthenticated_remote_hash_matches=True,local_original_hash_matches=True)
    with futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(one,selected))
    if any(file_sha256(root/p)!=sha for p,sha in binding.items()):raise AssertionError('Preserved evidence changed during verification')
    out.mkdir();report=dict(status='VERIFIED_CLOSED_RETRIEVABLE_WEB_DOWNLOAD_CACHE',registered_unix=started,
        absolute_deadline_unix=started+1200,rows=rows,preserved_evidence_sha256=binding,
        source_code_sha256=file_sha256(__file__),candidate_bytes=sum(r['bytes'] for r in rows),
        numeric_corpora_states_and_receipts_excluded=True,deleted_files=0,wall_seconds=time.time()-started,
        scope='Closed raw download cache only. Selected numeric corpora, SQLite, metadata, states and original receipts are preserved. Immutable public revisions/checksums permit exact refetch; no source contents are published.')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','preserved_evidence_sha256')}),flush=True)


def apply(root):
    out=root/IDENTITY;file=out/'verification.json';report=json.loads(file.read_text());cache=root/'natural-append-17000m-v3/raw'
    if (out/'applied.json').exists():raise FileExistsError('No silent maintenance reuse')
    if (report['status']!='VERIFIED_CLOSED_RETRIEVABLE_WEB_DOWNLOAD_CACHE' or report['source_code_sha256']!=file_sha256(__file__)
            or time.time()>report['absolute_deadline_unix']):raise ValueError('Fresh exact verified cache scope required')
    if any(file_sha256(root/p)!=sha for p,sha in report['preserved_evidence_sha256'].items()):raise AssertionError('Preserved evidence changed')
    before=shutil.disk_usage(root).free;deleted=[]
    # Revalidate all local fingerprints before the first unlink. No model,
    # optimizer, token array, SQLite or manifest path can pass the whitelist.
    for row in report['rows']:
        path,receipt,info=validate_candidate(cache,dict(path=str(cache/row['file']),domain='web',receipt=row['receipt']))
        if (info.st_ino,info.st_dev,info.st_size)!=(row['inode'],row['device'],row['bytes']) or file_sha256(path)!=receipt['sha256']:
            raise AssertionError('Verified cache changed before release')
    for row in report['rows']:
        if time.time()>report['absolute_deadline_unix']:raise TimeoutError('Original closed-cache maintenance clock')
        (cache/row['file']).unlink();deleted.append(row['file'])
        (out/'progress.json').write_text(json.dumps(dict(deleted_files=deleted))+'\n')
    if any(file_sha256(root/p)!=sha for p,sha in report['preserved_evidence_sha256'].items()):raise AssertionError('Preserved metadata changed')
    result=dict(status='RELEASED_VERIFIED_PUBLIC_DOWNLOAD_CACHE',verification_sha256=file_sha256(file),
        deleted_files=deleted,deleted_bytes=sum(r['bytes'] for r in report['rows']),free_bytes_before=before,
        free_bytes_after=shutil.disk_usage(root).free,preserved_evidence_sha256=report['preserved_evidence_sha256'],
        numeric_corpora_states_and_receipts_untouched=True,research_examples=0,optimizer_updates=0,assigned_chips=0,
        applied_unix=time.time(),scope=report['scope'])
    (out/'applied.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    apply(args.root) if args.apply else verify(args.root)
