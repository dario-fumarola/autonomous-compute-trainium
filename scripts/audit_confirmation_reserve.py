"""Independent SQL, source-row/token and reconstruction audit; no model scoring."""
import argparse
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from tokenizers import Tokenizer

from compute_trainium.data import content_hash
from compute_trainium.training import file_sha256
from scripts.prepare_learning_corpus import reserve
from scripts.prepare_natural import rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--generations',nargs='+',type=Path,required=True)
    p.add_argument('--heldout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args();started=time.time()
    def guard():
        reserve(a.output)
        if time.time()>a.deadline_unix:raise TimeoutError('Independent fresh-reserve audit deadline')
    guard();plan=json.loads((a.output/'plan.json').read_text());manifest=json.loads((a.output/'manifest.json').read_text())
    if manifest['status']!='PASS_FRESH_CONFIRMATION_RESERVE_PREPARATION_AUDIT_PENDING':raise ValueError('Preparation incomplete')
    if file_sha256(a.output/'plan.json')!=manifest['plan_sha256']:raise AssertionError('Preparation plan changed')
    if [file_sha256(g/'manifest.json') for g in a.generations]!=plan['generation_manifest_sha256']:raise AssertionError('Generation membership changed')
    documents=[]
    for name,field in [('dev.json','dev_sha256'),('test-sealed.json','test_sha256')]:
        if file_sha256(a.output/name)!=manifest[field]:raise AssertionError('Panel file changed')
        documents+=json.loads((a.output/name).read_text())
    db=sqlite3.connect(':memory:');db.execute('create table fresh(id text primary key,source_group text,role text,domain text)')
    db.executemany('insert into fresh values(?,?,?,?)',[(d['id'],d['group'],d['role'],d['domain']) for d in documents])
    db.execute('create index fresh_group on fresh(source_group)')
    old_ids=set();old_groups=set()
    for name,sha in zip(['dev.json','test-sealed.json'],plan['protected_panel_sha256']):
        if file_sha256(a.heldout/name)!=sha:raise AssertionError('Protected prior panel changed')
        for doc in json.loads((a.heldout/name).read_text()):old_ids.add(doc['id']);old_groups.add(doc['group'])
    remaining=set(old_ids)
    db.execute('create table protected_ids(id text primary key)')
    db.executemany('insert into protected_ids values(?)',[(key,) for key in old_ids]);db.commit()
    if db.execute('select count(*) from (select source_group from fresh group by source_group having count(distinct role)>1)').fetchone()[0]:
        raise AssertionError('Cross-role source group overlap')
    reconstructed=0
    for index,g in enumerate(a.generations):
        guard();path=g/('documents.sqlite' if index==0 else 'extension/documents.sqlite')
        if file_sha256(path)!=plan['prior_database_sha256'][index]:raise AssertionError('Prior accepted database changed')
        alias=f'old{index}';db.execute(f'attach database ? as {alias}',(str(path),))
        if db.execute(f'select count(*) from fresh f join {alias}.documents p on f.id=p.id').fetchone()[0]:raise AssertionError('Prior exact ID overlap')
        if db.execute(f'select count(*) from {alias}.documents p not indexed cross join fresh f indexed by fresh_group on p.source_group=f.source_group').fetchone()[0]:raise AssertionError('Prior source group overlap')
        remaining.difference_update(x[0] for x in db.execute(f'select p.id from protected_ids p join {alias}.documents o on p.id=o.id'))
        mapped=np.load(a.output/'index'/f'ids-{index:02}.npy',mmap_mode='r')
        cursor=db.execute(f'select id from {alias}.documents order by id');offset=0
        while batch:=cursor.fetchmany(32768):
            values=np.asarray([x[0] for x in batch],dtype='S64')
            if not np.array_equal(mapped[offset:offset+len(values)],values):raise AssertionError('Mapped IDs differ from accepted metadata')
            offset+=len(values)
        found=np.load(a.output/'index'/f'found-{index:02}.npy',mmap_mode='r')
        if offset!=len(mapped) or found.shape!=mapped.shape or found.dtype!=bool or not np.all(found):raise AssertionError('Incomplete raw sketch reconstruction coverage')
        reconstructed+=offset;db.execute(f'detach database {alias}')
    if old_ids & {d['id'] for d in documents} or old_groups & {d['group'] for d in documents}:raise AssertionError('Old panel overlap')
    extra=len(a.generations)
    mapped=np.load(a.output/'index'/f'ids-{extra:02}.npy',mmap_mode='r')
    if not np.array_equal(mapped,np.asarray(sorted(remaining),dtype='S64')):raise AssertionError('Protected-panel extra IDs differ')
    found=np.load(a.output/'index'/f'found-{extra:02}.npy',mmap_mode='r')
    if not np.all(found):raise AssertionError('Protected-panel raw reconstruction incomplete')
    reconstructed+=len(found)
    if reconstructed!=manifest['old_documents_reconstructed']:raise AssertionError('Coverage count differs')
    mapping=json.loads((a.output/'raw-source-map.json').read_text());sources=defaultdict(list)
    for entry in mapping:sources[entry['domain']].append(entry)
    tokenizer_path=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    if file_sha256(tokenizer_path)!=plan['tokenizer_sha256']:raise AssertionError('Tokenizer changed')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));wanted=defaultdict(dict)
    for doc in documents:
        if doc['split']!=('test' if doc['role']=='test' else 'dev') or doc['cohort']!=(None if doc['role']=='test' else doc['role']):raise AssertionError('Incorrect panel role')
        wanted[doc['domain'],doc['shard']][doc['raw_row']]=doc
    checked=0
    for (domain,shard),positions in sorted(wanted.items()):
        guard();entry=sources[domain][shard];path=Path(entry['path'])
        if file_sha256(path)!=entry['receipt']['sha256']:raise AssertionError('Chosen raw shard changed')
        for row_number,row in enumerate(rows(path)):
            if row_number not in positions:continue
            doc=positions[row_number];text=row.get('text',row.get('content',''))
            source=row.get('repo_name','') if domain=='code' else row.get('url','')
            plain=('repo:'+source.strip().lower().removesuffix('.git') if domain=='code'
                   else 'host:'+(urlsplit(source).hostname or '').lower().removeprefix('www.'))
            group=hashlib.sha256(plain.encode()).hexdigest()
            natural_bucket=int(hashlib.sha256(('natural-v1:'+plain).encode()).hexdigest()[:16],16)%100
            role=['tune','curve','test'][int(hashlib.sha256(('confirmation-reserve-v1:'+group).encode()).hexdigest()[:16],16)%3]
            encoded=tokenizer.encode(text,add_special_tokens=False).ids
            if content_hash(text)!=doc['id'] or group!=doc['group'] or role!=doc['role'] or natural_bucket<80:
                raise AssertionError('Raw content/group/role differs')
            if source!=doc['source'] or encoded!=doc['tokens'] or not 64<=len(encoded)<=8192 or len(text)>100000:
                raise AssertionError('Raw target tokens or source filters differ')
            if domain=='code' and (row.get('autogenerated',False) or str(row.get('license','')).lower() not in
                    {'mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','unlicense','cc0-1.0'}):raise AssertionError('Code policy mismatch')
            if int(doc['id'][:8],16)%8:raise AssertionError('Content subsampling differs')
            checked+=1;del positions[row_number]
            if not positions:break
        if positions:raise AssertionError('Declared raw rows missing')
    counts={}
    for domain in ['web','math','code']:
        for role in ['tune','curve','test']:
            subset=[d for d in documents if d['domain']==domain and d['role']==role]
            count={'documents':len(subset),'real_targets':sum(len(d['tokens']) for d in subset),'source_groups':len({d['group'] for d in subset})}
            if count!=manifest['counts'][domain+'/'+role] or count['real_targets']<250000 or count['source_groups']<100:raise AssertionError('Declared panel quota differs')
            if any(len([d for d in subset if d['group']==group])>2 for group in {d['group'] for d in subset}):raise AssertionError('Group document cap exceeded')
            counts[domain+'/'+role]=count
    guard();report={'status':'PASS_INDEPENDENT_FRESH_RESERVE_DATA_AUDIT','manifest_sha256':file_sha256(a.output/'manifest.json'),
        'auditor_sha256':file_sha256(__file__),'old_documents_reconstructed':reconstructed,'every_selected_raw_row_and_token_verified':checked,
        'mapped_ids_match_all_prior_databases':True,'all_raw_sketch_coverage_bits_present':True,'prior_exact_id_overlap':0,
        'prior_source_group_overlap':0,'cross_role_source_group_overlap':0,'counts':counts,'wall_seconds':time.time()-started,
        'test_model_scoring_performed':False,'terminal_attempt_registered':False,
        'scope':'Independent SQL identity/group checks over every corpus generation and prior panels, mapped-index export equality/coverage, and raw-row/complete-token reconstruction for every selected document. Does not independently repeat every minhash comparison; that approximate screening uses the separately tested inverted algorithm and complete prior-ID raw coverage. Data audit only, no scientific confirmation or alpha spending.'}
    (a.output/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
