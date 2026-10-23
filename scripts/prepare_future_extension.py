"""Bounded second natural-data append; never changes the running corpus.

Rebuild duplicate sketches for both accepted corpus generations and the newer
held-out panels before admitting TRAIN. All content stays in private storage.
This prepares data only; checkpoint migration and future training need their
own protocol and qualification.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np

from compute_trainium.corpus_audit import NearDuplicateIndex, text_sketch
from compute_trainium.data import content_hash, source_group, split_for_group
from compute_trainium.training import file_sha256
from scripts.prepare_extension_corpus import concatenate
from scripts.prepare_learning_corpus import (bounded_map, freeze_packing, interleaved_batches,
                                             process_batch, reserve, worker_init)
from scripts.prepare_natural import SOURCES, download


def initialize(tokenizer, ids, protected_groups):
    global PRIOR_IDS, PROTECTED_GROUPS
    PRIOR_IDS, PROTECTED_GROUPS = ids, protected_groups
    worker_init(tokenizer)


def inspect_old(batch):
    output=[]
    for domain, shard, row in batch:
        text=row.get('text',row.get('content',''))
        if text and len(text)<=100_000:
            identity=content_hash(text)
            if identity in PRIOR_IDS:
                signature,sketch=text_sketch(text);output.append((identity,signature,sketch))
    return output


def candidate_batch(batch):
    output=[]
    for domain, shard, row in batch:
        text=row.get('text',row.get('content',''))
        if not text or len(text)>100_000:output.append({'reject':'text_length_filter'});continue
        if content_hash(text) in PRIOR_IDS:output.append({'reject':'prior_document'});continue
        try:
            group=source_group(domain,row.get('repo_name','') if domain=='code' else row.get('url',''))
        except ValueError:output.append({'reject':'missing_source'});continue
        if split_for_group(group)!='train' or hashlib.sha256(group.encode()).hexdigest() in PROTECTED_GROUPS:
            output.append({'reject':'held_out_source_group'});continue
        output.extend(process_batch([(domain,shard,row)]))
    return output


def prior_identities(databases, heldout_paths):
    identities=set();groups=set();train_groups=set();by_domain={d:set() for d in SOURCES}
    for path in databases:
        db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        for identity,domain,group,split in db.execute('select id,domain,source_group,split from documents'):
            identities.add(identity);by_domain[domain].add(identity)
            if split!='train':groups.add(group)
            else:train_groups.add(group)
        db.close()
    for path in heldout_paths:
        for doc in json.loads(path.read_text()):
            if doc['split']=='train':raise ValueError('Protected panel contains TRAIN')
            identities.add(doc['id']);by_domain[doc['domain']].add(doc['id']);groups.add(doc['group'])
    if train_groups & groups:raise ValueError('Prior TRAIN intersects protected source groups')
    return identities,groups,by_domain


def run(a):
    started=time.time();reserve(a.output)
    def guard():
        reserve(a.output)
        if time.time()>a.deadline_unix:raise TimeoutError('Bounded data preparation deadline')
    parent=json.loads((a.parent/'manifest.json').read_text());legacy=json.loads((a.legacy/'manifest.json').read_text())
    protected=[a.heldout/'dev.json',a.heldout/'test-sealed.json']
    tokenizer=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    plan={'identity':'future-natural-append-v2','purpose':'DATA_PREPARATION_ONLY_NO_TRAINING',
          'synthetic_generation':False,'started_unix':started,'deadline_unix':a.deadline_unix,
          'additional_target_quota':a.additional_tokens,'requested_shards':{'web':14,'math':24,'code':24},
          'workers':a.workers,'parent_manifest_sha256':file_sha256(a.parent/'manifest.json'),
          'parent_packing_manifest_sha256':file_sha256(a.parent/'packed/manifest.json'),
          'prior_database_sha256':[file_sha256(p) for p in [a.legacy/'documents.sqlite',a.parent/'extension/documents.sqlite']],
          'tokenizer_sha256':file_sha256(tokenizer),
          'protected_panel_sha256':[file_sha256(p) for p in protected],
          'source_sha256':{str(p.resolve().relative_to(Path.cwd())):file_sha256(p) for p in [Path(__file__),
              Path('scripts/prepare_extension_corpus.py'),Path('scripts/prepare_learning_corpus.py'),
              Path('scripts/prepare_natural.py'),Path('compute_trainium/corpus_audit.py'),Path('compute_trainium/data.py')]}}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (a.output/'raw').mkdir();extension=a.output/'extension';extension.mkdir()
    ids,groups,domain_ids=prior_identities([a.legacy/'documents.sqlite',a.parent/'extension/documents.sqlite'],protected)
    print(json.dumps({'event':'prior_sets','documents':len(ids),'protected_groups':len(groups)}),flush=True)
    def fetch(item):
        guard();domain,index=item;repo,revision,name=SOURCES[domain]
        name=name.replace('train-00000-',f'train-{index:05d}-') if domain!='code' else name.replace('000000000001',f'{index+1:012d}')
        suffix='.json.gz' if domain=='code' else '.parquet'
        if index==0:path=tokenizer.parent/(domain+suffix)
        elif index<len(legacy['sources'][domain]):path=a.legacy/(domain+f'-{index}'+suffix)
        elif index<len(parent['sources'][domain]):path=a.parent/'raw'/(domain+f'-{index}'+suffix)
        else:path=a.output/'raw'/(domain+f'-{index}'+suffix)
        sha=download(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}',path);guard()
        if index<len(parent['sources'][domain]) and sha!=parent['sources'][domain][index]['sha256']:
            raise ValueError('Pinned source changed')
        return domain,index,path,{'repository':repo,'revision':revision,'file':name,'sha256':sha}
    sources={d:[] for d in SOURCES}
    with cf.ThreadPoolExecutor(3) as pool:
        for domain,index,path,receipt in pool.map(fetch,[(d,i) for d,n in plan['requested_shards'].items() for i in range(n)]):
            sources[domain].append((index,path,receipt))
    near=NearDuplicateIndex();found=set();seen=set(ids);rejected=Counter();train_counts={}
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    with cf.ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,
                               initargs=(str(tokenizer),ids,groups)) as pool:
        for domain in SOURCES:
            paths=[p for i,p,_ in sources[domain] if i<len(parent['sources'][domain])]
            for batch in bounded_map(pool,inspect_old,interleaved_batches(domain,paths),a.workers*2):
                guard()
                for identity,signature,sketch in batch:
                    if identity not in found:near.add(signature,sketch);found.add(identity)
                if domain_ids[domain]<=found:break
            if not domain_ids[domain]<=found:raise ValueError('Incomplete prior raw-text index: '+domain)
            print(json.dumps({'event':'index_rebuilt','domain':domain,'documents':len(found),'seconds':time.time()-started}),flush=True)
        if found!=ids:raise AssertionError('Prior duplicate reconstruction incomplete')
        db=sqlite3.connect(extension/'documents.sqlite')
        db.execute('create table documents(id text primary key,domain text,source_group text,split text,cohort text,offset integer,length integer,source text,license text,shard integer)')
        for domain,weight in [('web',.6),('math',.2),('code',.2)]:
            quota=int(a.additional_tokens*weight);accepted=offset=0;progress=-1
            with (extension/(domain+'.u16')).open('wb') as stream:
                for batch in bounded_map(pool,candidate_batch,interleaved_batches(domain,[p for _,p,_ in sources[domain]]),a.workers*2):
                    guard()
                    for doc in batch:
                        if 'reject' in doc:rejected[domain+'/'+doc['reject']]+=1;continue
                        if accepted>=quota:break
                        if doc['id'] in seen:rejected[domain+'/exact_duplicate']+=1;continue
                        if doc['split']!='train' or doc['group'] in groups:raise AssertionError('Protected source group admitted')
                        reason=near.check(doc['signature'],doc['sketch'])
                        if reason:rejected[domain+'/'+reason]+=1;continue
                        seen.add(doc['id']);near.add(doc.pop('signature'),doc.pop('sketch'))
                        length=len(doc['tokens']);np.asarray(doc['tokens'],dtype=np.uint16).tofile(stream)
                        db.execute('insert into documents values(?,?,?,?,?,?,?,?,?,?)',
                            (doc['id'],domain,doc['group'],'train',None,offset,length,doc['source'],doc['license'],doc['shard']))
                        accepted+=length;offset+=length
                    db.commit()
                    if accepted//100_000_000!=progress:
                        progress=accepted//100_000_000
                        print(json.dumps({'event':'new_train','domain':domain,'targets':accepted,'seconds':time.time()-started}),flush=True)
                    if accepted>=quota:break
            if accepted<quota:raise ValueError(f'Insufficient natural {domain}: {accepted} < {quota}')
            train_counts[domain]=accepted
        guard();extension_packing=freeze_packing(extension,db,train_counts,1024);db.close()
    guard();packing=concatenate(a.parent,extension,a.output);guard()
    for name in ['dev.json','test-sealed.json']:shutil.copy2(a.heldout/name,a.output/name)
    report={'status':'PASS_FROZEN_SECOND_NATURAL_EXTENSION','synthetic_generation':False,
            'plan_sha256':file_sha256(a.output/'plan.json'),'parent_manifest_sha256':plan['parent_manifest_sha256'],
            'sources':{d:[s for _,_,s in values] for d,values in sources.items()},
            'prior_and_fresh_panel_documents_reconstructed':len(found),'duplicate_reconstruction_complete':found==ids,
            'protected_source_groups':len(groups),'heldout_source_group_overlap':0,
            'additional_domain_targets':train_counts,'rejected':dict(rejected),
            'near_duplicate_audit':legacy['near_duplicate_audit'],'packing':packing,'extension_packing':extension_packing,
            'dev_sha256':file_sha256(a.output/'dev.json'),'test_sha256':file_sha256(a.output/'test-sealed.json'),
            'protected_panels_unchanged':all(file_sha256(a.heldout/n)==file_sha256(a.output/n) for n in ['dev.json','test-sealed.json']),
            'preparation_wall_seconds':time.time()-started,
            'scope':'Data preparation only, no model selection/training/TEST scoring. Entire old packed prefix hash-verified unchanged. New source migration is not yet qualified. Existing panels are retained for provenance/exclusion, not reused to confirm a new intervention.'}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['sources','packing','extension_packing','rejected']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--legacy',type=Path,required=True);p.add_argument('--heldout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=20)
    p.add_argument('--additional-tokens',type=int,default=3_250_000_000)
    p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args();started=time.time()
    repo=Path(__file__).resolve().parents[1]
    if repo==a.output.resolve() or repo in a.output.resolve().parents:raise ValueError('Store corpus outside Git')
    if not 1<=a.workers<=48 or a.additional_tokens<5 or a.deadline_unix<started+300:
        raise ValueError('Invalid bounded preparation settings')
    a.output.mkdir(exist_ok=False)
    try:run(a)
    except Exception as error:
        if a.output.exists() and not (a.output/'manifest.json').exists():
            (a.output/'failure.json').write_text(json.dumps({'status':'STOPPED_DATA_PREPARATION',
                'error':repr(error),'seconds':time.time()-started},indent=2)+'\n')
        raise


if __name__=='__main__':main()
