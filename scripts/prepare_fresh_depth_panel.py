"""Fresh natural-source groups, screened against every earlier accepted document.

Uses only the already downloaded pinned shards. Reconstructs the complete old
raw-text near-duplicate index before admitting any new held-out example.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from compute_trainium.corpus_audit import NearDuplicateIndex, text_sketch
from compute_trainium.data import content_hash, source_group, split_for_group
from compute_trainium.training import file_sha256
from scripts.prepare_learning_corpus import bounded_map, interleaved_batches, process_batch, reserve, worker_init


def initialize(tokenizer, old_ids, old_groups):
    global OLD_IDS, OLD_GROUPS
    OLD_IDS, OLD_GROUPS = old_ids, old_groups
    worker_init(tokenizer)


def inspect_batch(batch):
    output=[]
    for domain,shard,row in batch:
        text=row.get('text',row.get('content',''))
        if not text or len(text)>100_000:continue
        identity=content_hash(text)
        if identity in OLD_IDS:
            signature,sketch=text_sketch(text)
            output.append({'old_id':identity,'signature':signature,'sketch':sketch})
            continue
        source=row.get('repo_name','') if domain=='code' else row.get('url','')
        try:
            group=source_group(domain,source)
        except ValueError:continue
        group_hash=hashlib.sha256(group.encode()).hexdigest()
        if split_for_group(group)=='train' or group_hash in OLD_GROUPS:continue
        # Fixed content-hash subsampling bounds CPU/data costs without looking at loss.
        if int(identity[:8],16)%4:continue
        for doc in process_batch([(domain,shard,row)]):
            if 'reject' in doc:continue
            role=['tune','curve','test'][int(hashlib.sha256(('depth-mixture-v1:'+group).encode()).hexdigest()[:16],16)%3]
            doc['split']='test' if role=='test' else 'dev'
            doc['cohort']=None if role=='test' else role
            doc['role']=role
            output.append(doc)
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--extension',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=20);p.add_argument('--targets-per-domain-role',type=int,default=250000)
    a=p.parse_args();a.output.mkdir(exist_ok=False);reserve(a.output);started=time.monotonic()
    ids=set();groups=set();domain_ids=defaultdict(set)
    for db_path in [a.parent/'documents.sqlite',a.extension/'extension/documents.sqlite']:
        db=sqlite3.connect(f'file:{db_path}?mode=ro',uri=True)
        for identity,domain,group in db.execute('select id,domain,source_group from documents'):
            ids.add(identity);groups.add(group);domain_ids[domain].add(identity)
        db.close()
    print(json.dumps({'event':'old_identity_sets','documents':len(ids),'source_groups':len(groups)}),flush=True)
    original=json.loads((a.parent/'manifest.json').read_text())
    extended=json.loads((a.extension/'manifest.json').read_text())
    tokenizer=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    paths={};source_receipts={}
    for domain,sources in extended['sources'].items():
        paths[domain]=[];source_receipts[domain]=sources
        for i,source in enumerate(sources):
            suffix='.json.gz' if domain=='code' else '.parquet'
            path=(tokenizer.parent/(domain+suffix) if i==0 else
                  a.parent/(domain+f'-{i}'+suffix) if i<len(original['sources'][domain]) else
                  a.extension/'raw'/(domain+f'-{i}'+suffix))
            if file_sha256(path)!=source['sha256']:raise ValueError('Pinned shard changed')
            paths[domain].append(path)
    near=NearDuplicateIndex();found=set();pending=defaultdict(list)
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    with cf.ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,
         initargs=(str(tokenizer),ids,groups)) as pool:
        for domain in ['web','math','code']:
            previous=0
            for batch in bounded_map(pool,inspect_batch,interleaved_batches(domain,paths[domain]),a.workers*2):
                reserve(a.output)
                for doc in batch:
                    if 'old_id' in doc:
                        if doc['old_id'] not in found:
                            near.add(doc['signature'],doc['sketch']);found.add(doc['old_id'])
                    elif len(pending[domain,doc['role']])<3000:
                        pending[domain,doc['role']].append(doc)
                if len(found)//100000!=previous:
                    previous=len(found)//100000
                    print(json.dumps({'event':'rebuild','domain':domain,'found':len(found),'seconds':time.monotonic()-started}),flush=True)
            if not domain_ids[domain]<=found:raise ValueError('Old duplicate index incomplete: '+domain)
    if found!=ids:raise AssertionError('Old document reconstruction incomplete')
    chosen=[];counts={};rejected=Counter();seen=set(ids);role_groups={}
    for domain in ['web','math','code']:
        for role in ['tune','curve','test']:
            total=0
            for doc in pending[domain,role]:
                if doc['id'] in seen:rejected['exact_duplicate']+=1;continue
                if doc['group'] in groups:raise AssertionError('Old source group leaked')
                if doc['group'] in role_groups and role_groups[doc['group']]!=role:raise AssertionError('Cross-role source group')
                reason=near.check(doc['signature'],doc['sketch'])
                if reason:rejected[reason]+=1;continue
                near.add(doc.pop('signature'),doc.pop('sketch'));seen.add(doc['id'])
                role_groups[doc['group']]=role
                total+=len(doc['tokens']);doc.pop('role');chosen.append(doc)
                if total>=a.targets_per_domain_role:break
            if total<a.targets_per_domain_role:raise ValueError(f'Insufficient fresh {domain}/{role}: {total}')
            counts[domain+'/'+role]={'real_targets':total,'documents':sum(d['domain']==domain and (('test' if d['split']=='test' else d['cohort'])==role) for d in chosen)}
    for split,name in [('dev','dev.json'),('test','test-sealed.json')]:
        path=a.output/name;path.write_text(json.dumps([d for d in chosen if d['split']==split]));path.chmod(0o600)
    manifest={'status':'PASS_FRESH_NATURAL_DEPTH_PANEL','synthetic_generation':False,
       'selection':'Unused original-heldout source groups, excluding all old accepted source groups; content hash mod4=0; independent depth-mixture-v1 group hash assigns tune/curve/test.',
       'counts':counts,'sources':source_receipts,'old_documents_reconstructed':len(found),'old_source_groups_excluded':len(groups),
       'near_duplicate_audit':original['near_duplicate_audit'],'rejected':dict(rejected),
       'tokenizer_sha256':file_sha256(tokenizer),'dev_sha256':file_sha256(a.output/'dev.json'),
       'test_sha256':file_sha256(a.output/'test-sealed.json'),'source_group_overlap_with_old':0,
       'cross_role_source_group_overlap':0,'preparation_seconds':time.monotonic()-started}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['sources']}),flush=True)


if __name__=='__main__':main()
