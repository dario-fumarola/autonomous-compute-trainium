"""Bounded natural-data append across an explicitly verified corpus lineage.

Data only: no model training, evaluation or TEST-based choices. Preserve every
parent packed position and reconstruct all prior duplicate sketches first.
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

from compute_trainium.corpus_audit import NearDuplicateIndex
from compute_trainium.training import file_sha256
from scripts import prepare_future_extension as previous
from scripts.prepare_extension_corpus import concatenate
from scripts.prepare_learning_corpus import bounded_map, freeze_packing, interleaved_batches, reserve
from scripts.prepare_natural import SOURCES, download


def lineage(generations):
    if not generations or len(set(p.resolve() for p in generations))!=len(generations):
        raise ValueError('Explicit ordered, non-repeated corpus generations required')
    manifests=[];databases=[]
    for index,root in enumerate(generations):
        manifest=json.loads((root/'manifest.json').read_text())
        if not manifest['status'].startswith('PASS_FROZEN_'):
            raise ValueError('Unqualified parent corpus')
        if index:
            if manifest['parent_manifest_sha256']!=file_sha256(generations[index-1]/'manifest.json'):
                raise ValueError('Missing or reordered corpus generation')
            source=json.loads((root/'source.json').read_text())
            if source['parent_source_sha256']!=file_sha256(generations[index-1]/'source.json'):
                raise ValueError('Source lineage differs')
        database=root/('documents.sqlite' if index==0 else 'extension/documents.sqlite')
        if not database.is_file():raise ValueError('Missing accepted-document database')
        manifests.append(manifest);databases.append(database)
    return manifests,databases


def source_key(domain,receipt):
    return (domain,receipt['repository'],receipt['revision'],receipt['file'])


def old_raw_paths(generations,manifests,tokenizer):
    """Find by immutable receipt, never assume index means the same crawl."""
    mapping={};hashes={}
    def add(domain,receipt,choices):
        key=source_key(domain,receipt)
        if key in mapping:
            if mapping[key][1]['sha256']!=receipt['sha256']:raise ValueError('Conflicting source receipt')
            return
        for path in choices:
            if not path.is_file():continue
            resolved=path.resolve()
            if resolved not in hashes:hashes[resolved]=file_sha256(resolved)
            if hashes[resolved]==receipt['sha256']:
                mapping[key]=(resolved,receipt);return
        raise ValueError('Cannot locate checksum-matched prior raw source: '+receipt['file'])
    for root,manifest in zip(generations,manifests):
        private=root/'raw-source-map.json'
        if private.exists():
            for entry in json.loads(private.read_text()):
                add(entry['domain'],entry['receipt'],[Path(entry['path'])])
        for domain,receipts in manifest['sources'].items():
            suffix='.json.gz' if domain=='code' else '.parquet'
            for index,receipt in enumerate(receipts):
                add(domain,receipt,[tokenizer.parent/(domain+suffix),root/(domain+f'-{index}'+suffix),
                                    root/'raw'/(domain+f'-{index}'+suffix)])
    return mapping


def listed_files(domain,directory):
    repository,revision,_=SOURCES[domain]
    url=f'https://huggingface.co/api/datasets/{repository}/tree/{revision}/{directory}'
    response=requests.get(url,params={'limit':1000},timeout=60);response.raise_for_status()
    if response.links.get('next'):raise ValueError('Source listing is paginated; refuse silent omission')
    suffix='.json.gz' if domain=='code' else '.parquet'
    files=sorted(row['path'] for row in response.json() if row['type']=='file' and row['path'].endswith(suffix))
    if not files:raise ValueError('No eligible immutable source files')
    return [{'repository':repository,'revision':revision,'file':name} for name in files]


def run(a):
    started=time.time();reserve(a.output)
    def guard():
        reserve(a.output)
        if time.time()>a.deadline_unix:raise TimeoutError('Bounded chained data preparation deadline')
    manifests,databases=lineage(a.generations);parent=a.generations[-1]
    protected=[a.heldout/'dev.json',a.heldout/'test-sealed.json']
    tokenizer=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    tokenizer_hash=file_sha256(tokenizer)
    parent_contract=json.loads((parent/'plan.json').read_text()) if (parent/'plan.json').exists() else manifests[-1]
    if parent_contract.get('tokenizer_sha256')!=tokenizer_hash:raise ValueError('Pinned parent tokenizer differs or is undeclared')
    prior=old_raw_paths(a.generations,manifests,tokenizer);guard()
    requested={}
    for domain,directories in [('web',['data/CC-MAIN-2013-48','data/CC-MAIN-2014-10']),
                               ('math',['finemath-4plus']),('code',[''])]:
        receipts=list(manifests[-1]['sources'][domain]);known={source_key(domain,r) for r in receipts}
        for directory in directories:
            guard()
            for receipt in listed_files(domain,directory):
                key=source_key(domain,receipt)
                if key not in known:receipts.append(receipt);known.add(key)
        requested[domain]=receipts
    plan={'identity':'future-natural-append-v3','purpose':'DATA_PREPARATION_ONLY_NO_TRAINING',
        'synthetic_generation':False,'started_unix':started,'deadline_unix':a.deadline_unix,
        'additional_target_quota':a.additional_tokens,'workers':a.workers,
        'parent_manifest_sha256':file_sha256(parent/'manifest.json'),
        'generation_manifest_sha256':[file_sha256(p/'manifest.json') for p in a.generations],
        'prior_database_sha256':[file_sha256(p) for p in databases],
        'tokenizer_sha256':tokenizer_hash,'protected_panel_sha256':[file_sha256(p) for p in protected],
        'requested_sources':requested,'source_sha256':{str(p):file_sha256(p) for p in [
            Path('scripts/prepare_chained_extension.py'),Path('scripts/audit_chained_extension.py'),Path('scripts/prepare_future_extension.py'),
            Path('scripts/prepare_extension_corpus.py'),Path('scripts/prepare_learning_corpus.py'),
            Path('scripts/prepare_natural.py'),Path('compute_trainium/corpus_audit.py'),Path('compute_trainium/data.py')]}}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (a.output/'raw').mkdir();extension=a.output/'extension';extension.mkdir()
    def fetch(item):
        domain,index,receipt=item;guard();key=source_key(domain,receipt)
        if key in prior:path,receipt=prior[key]
        else:
            suffix='.json.gz' if domain=='code' else '.parquet';path=a.output/'raw'/(domain+f'-{index}'+suffix)
            receipt=dict(receipt,sha256=download(f"https://huggingface.co/datasets/{receipt['repository']}/resolve/{receipt['revision']}/{receipt['file']}",path))
        guard();return domain,index,path,receipt
    sources={domain:[] for domain in SOURCES}
    with cf.ThreadPoolExecutor(3) as pool:
        for domain,index,path,receipt in pool.map(fetch,[(d,i,r) for d,rows in requested.items() for i,r in enumerate(rows)]):
            sources[domain].append((index,path,receipt))
    (a.output/'raw-source-map.json').write_text(json.dumps([
        {'domain':d,'path':str(path),'receipt':receipt} for d,rows in sources.items() for _,path,receipt in rows],indent=2)+'\n')
    ids,groups,by_domain=previous.prior_identities(databases,protected)
    print(json.dumps({'event':'prior_sets','documents':len(ids),'protected_groups':len(groups)}),flush=True)
    near=NearDuplicateIndex();found=set();seen=set(ids);rejected=Counter();train_counts={}
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    with cf.ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=previous.initialize,
                               initargs=(str(tokenizer),ids,groups)) as pool:
        for domain in SOURCES:
            paths=[p for _,p,receipt in sources[domain] if source_key(domain,receipt) in prior]
            for batch in bounded_map(pool,previous.inspect_old,interleaved_batches(domain,paths),a.workers*2):
                guard()
                for identity,signature,sketch in batch:
                    if identity not in found:near.add(signature,sketch);found.add(identity)
                if by_domain[domain]<=found:break
            if not by_domain[domain]<=found:raise ValueError('Incomplete prior raw-text reconstruction: '+domain)
            print(json.dumps({'event':'index_rebuilt','domain':domain,'documents':len(found),'seconds':time.time()-started}),flush=True)
        if found!=ids:raise AssertionError('Prior duplicate reconstruction incomplete')
        db=sqlite3.connect(extension/'documents.sqlite')
        db.execute('create table documents(id text primary key,domain text,source_group text,split text,cohort text,offset integer,length integer,source text,license text,shard integer)')
        for domain,weight in [('web',.6),('math',.2),('code',.2)]:
            quota=int(a.additional_tokens*weight);accepted=offset=0;progress=-1
            with (extension/(domain+'.u16')).open('wb') as stream:
                for batch in bounded_map(pool,previous.candidate_batch,interleaved_batches(domain,[p for _,p,_ in sources[domain]]),a.workers*2):
                    guard()
                    for doc in batch:
                        if 'reject' in doc:rejected[domain+'/'+doc['reject']]+=1;continue
                        if accepted>=quota:break
                        if doc['id'] in seen:rejected[domain+'/exact_duplicate']+=1;continue
                        if doc['split']!='train' or doc['group'] in groups:raise AssertionError('Protected source admitted')
                        reason=near.check(doc['signature'],doc['sketch'])
                        if reason:rejected[domain+'/'+reason]+=1;continue
                        seen.add(doc['id']);near.add(doc.pop('signature'),doc.pop('sketch'))
                        length=len(doc['tokens']);np.asarray(doc['tokens'],dtype=np.uint16).tofile(stream)
                        db.execute('insert into documents values(?,?,?,?,?,?,?,?,?,?)',
                            (doc['id'],domain,doc['group'],'train',None,offset,length,doc['source'],doc['license'],doc['shard']))
                        accepted+=length;offset+=length
                    db.commit()
                    if accepted//200_000_000!=progress:
                        progress=accepted//200_000_000
                        print(json.dumps({'event':'new_train','domain':domain,'targets':accepted,'seconds':time.time()-started}),flush=True)
                    if accepted>=quota:break
            if accepted<quota:raise ValueError(f'Insufficient natural {domain}: {accepted} < {quota}')
            train_counts[domain]=accepted
        guard();extension_packing=freeze_packing(extension,db,train_counts,1024);db.close()
    guard();packing=concatenate(parent,extension,a.output);guard()
    for name in ['dev.json','test-sealed.json']:shutil.copy2(a.heldout/name,a.output/name)
    report={'status':'PASS_FROZEN_CHAINED_NATURAL_EXTENSION','synthetic_generation':False,
        'plan_sha256':file_sha256(a.output/'plan.json'),'parent_manifest_sha256':plan['parent_manifest_sha256'],
        'sources':{d:[r for _,_,r in rows] for d,rows in sources.items()},
        'prior_and_protected_documents_reconstructed':len(found),'duplicate_reconstruction_complete':found==ids,
        'protected_source_groups':len(groups),'additional_domain_targets':train_counts,'rejected':dict(rejected),
        'near_duplicate_audit':manifests[0]['near_duplicate_audit'],'packing':packing,'extension_packing':extension_packing,
        'dev_sha256':file_sha256(a.output/'dev.json'),'test_sha256':file_sha256(a.output/'test-sealed.json'),
        'protected_panels_unchanged':all(file_sha256(a.heldout/n)==file_sha256(a.output/n) for n in ['dev.json','test-sealed.json']),
        'preparation_wall_seconds':time.time()-started,
        'scope':'Data preparation only. New web crawls at the existing pinned revision expand the natural reserve; both matched arms must consume the same declared stream. Entire parent prefix unchanged. Approximate lexical duplicate screening is not a semantic guarantee. Existing TEST is retained only for provenance/exclusion and cannot confirm a new intervention. No model-source migration qualified.'}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'real_targets':packing['real_targets'],'seconds':report['preparation_wall_seconds']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--generations',nargs='+',type=Path,required=True)
    p.add_argument('--heldout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--additional-tokens',type=int,default=10_500_000_000)
    p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args();started=time.time()
    repo=Path(__file__).resolve().parents[1]
    if repo==a.output.resolve() or repo in a.output.resolve().parents:raise ValueError('Private corpus must stay outside Git')
    if not 1<=a.workers<=20 or a.additional_tokens<5 or a.deadline_unix<started+300:raise ValueError('Invalid bounded preparation')
    a.output.mkdir(exist_ok=False)
    try:run(a)
    except Exception as error:
        if not (a.output/'manifest.json').exists():
            (a.output/'failure.json').write_text(json.dumps({'status':'STOPPED_CHAINED_DATA_PREPARATION','error':repr(error),'seconds':time.time()-started},indent=2)+'\n')
        raise


if __name__=='__main__':main()
