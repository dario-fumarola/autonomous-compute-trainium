"""Append new natural TRAIN documents while preserving the entire old stream.

Reconstruct the original lexical duplicate index from pinned raw source shards,
then screen every additional document against old TRAIN/DEV/TEST and new TRAIN.
No generated examples, tokenizer changes, old-data rewriting or epoch wrapping.
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
from scripts.prepare_learning_corpus import (bounded_map, freeze_packing, interleaved_batches,
                                             process_batch, reserve, worker_init)
from scripts.prepare_natural import SOURCES, download


def initialize(tokenizer,ids):
    global ORIGINAL_IDS
    ORIGINAL_IDS=ids
    worker_init(tokenizer)


def rebuild_batch(batch):
    output=[]
    for domain,shard,row in batch:
        text=row.get('text',row.get('content',''))
        if not text or len(text)>100_000:continue
        identity=content_hash(text)
        if identity in ORIGINAL_IDS:
            signature,sketch=text_sketch(text)
            output.append((identity,signature,sketch))
    return output


def extension_batch(batch):
    output=[]
    for domain,shard,row in batch:
        text=row.get('text',row.get('content',''))
        if not text or len(text)>100_000:
            output.append({'reject':'text_length_filter'});continue
        identity=content_hash(text)
        if identity in ORIGINAL_IDS:
            output.append({'reject':'original_document'});continue
        try:
            source=row.get('repo_name','') if domain=='code' else row.get('url','')
            if split_for_group(source_group(domain,source))!='train':
                output.append({'reject':'held_out_source_group'});continue
        except ValueError:
            output.append({'reject':'missing_source'});continue
        output.extend(process_batch([(domain,shard,row)]))
    return output


def concatenate(parent,extension,output):
    """The old final overlap token stays; new source targets start after it."""
    a=json.loads((parent/'packed/manifest.json').read_text())
    b=json.loads((extension/'packed/manifest.json').read_text())
    out=output/'packed';out.mkdir()
    windows=a['windows']+b['windows'];length=windows*1024+1
    for name,dtype in [('tokens',np.int32),('domains',np.uint8)]:
        left=np.load(parent/'packed'/(name+'.npy'),mmap_mode='r')
        right=np.load(extension/'packed'/(name+'.npy'),mmap_mode='r')
        dest=np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=dtype,shape=(length,))
        for first in range(0,len(left),1024*1024):
            end=min(len(left),first+1024*1024)
            dest[first:end]=left[first:end]
        begin=len(left)
        for first in range(1,len(right),1024*1024):
            block=right[first:first+1024*1024];dest[begin:begin+len(block)]=block;begin+=len(block)
        assert begin==length;dest.flush()
    left=np.load(parent/'packed/prefix_counts.npy',mmap_mode='r')
    right=np.load(extension/'packed/prefix_counts.npy',mmap_mode='r')
    counts=np.lib.format.open_memmap(out/'prefix_counts.npy',mode='w+',dtype=np.int64,shape=(windows+1,3))
    counts[:len(left)]=left;counts[len(left):]=right[1:]+left[-1];counts.flush()
    tokens=np.load(out/'tokens.npy',mmap_mode='r');domains=np.load(out/'domains.npy',mmap_mode='r')
    fingerprint=hashlib.sha256();prefix=hashlib.sha256()
    old_length=a['windows']*1024+1
    for array in [tokens,domains]:
        for first in range(0,len(array),1024*1024):fingerprint.update(array[first:first+1024*1024].tobytes())
        for first in range(0,old_length,1024*1024):prefix.update(array[first:min(old_length,first+1024*1024)].tobytes())
    assert prefix.hexdigest()==a['fingerprint']
    source={'schema':'natural-append-v1','synthetic_generation':False,'data_seed':23,'context':1024,
            'parent_source_sha256':a['source_sha256'],'parent_packing_sha256':a['fingerprint'],
            'extension_source_sha256':b['source_sha256'],'extension_packing_sha256':b['fingerprint'],
            'train_domain_targets':dict(zip(['web','math','code'],map(int,counts[-1]))),
            'order_fingerprint':hashlib.sha256((a['order_fingerprint']+':'+b['order_fingerprint']).encode()).hexdigest(),
            'order_fingerprint_definition':'SHA256(parent order digest + colon + extension order digest)',
            'boundary':'Preserve old terminal overlap/padding token; append extension tokens[1:]. First new target follows that old boundary token.'}
    (output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    metadata={'context':1024,'seed':23,'windows':windows,'real_targets':int(counts[-1].sum()),
              'domain_counts':source['train_domain_targets'],'fingerprint':fingerprint.hexdigest(),
              'order_fingerprint':source['order_fingerprint'],'source_sha256':file_sha256(output/'source.json'),
              'prefix_parent':{'fingerprint':a['fingerprint'],'source_sha256':a['source_sha256'],'windows':a['windows']},
              'files':{p.name:file_sha256(p) for p in out.glob('*.npy')}}
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--additional-tokens',type=int,default=2_200_000_000)
    p.add_argument('--web-shards',type=int,default=6);p.add_argument('--shards',type=int,default=18)
    p.add_argument('--resume-downloads-only',action='store_true')
    args=p.parse_args()
    if args.resume_downloads_only:
        if (args.output/'extension/documents.sqlite').exists() or (args.output/'source.json').exists():
            raise ValueError('Only incomplete downloads can resume here; retain accepted/frozen data')
        archive=args.output/'interrupted-downloads';archive.mkdir(exist_ok=True)
        for part in (args.output/'raw').glob('*.partial'):part.rename(archive/part.name)
    else:args.output.mkdir(exist_ok=False)
    reserve(args.output);started=time.monotonic()
    (args.output/'raw').mkdir(exist_ok=args.resume_downloads_only)
    extension=args.output/'extension';extension.mkdir(exist_ok=args.resume_downloads_only)
    old_manifest=json.loads((args.parent/'manifest.json').read_text())
    db_old=sqlite3.connect(f'file:{args.parent}/documents.sqlite?mode=ro',uri=True)
    old_ids={r[0] for r in db_old.execute('select id from documents')}
    domain_ids={d:{r[0] for r in db_old.execute('select id from documents where domain=?',(d,))} for d in SOURCES}
    tokenizer=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    old_base=tokenizer.parent
    sources={}
    def fetch(item):
        domain,index=item;repo,rev,name=SOURCES[domain]
        name=name.replace('train-00000-',f'train-{index:05d}-') if domain!='code' else name.replace('000000000001',f'{index+1:012d}')
        suffix='.json.gz' if domain=='code' else '.parquet'
        if index==0:path=old_base/(domain+suffix)
        elif index<len(old_manifest['sources'][domain]):path=args.parent/(domain+f'-{index}'+suffix)
        else:path=args.output/'raw'/(domain+f'-{index}'+suffix)
        digest=download(f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}',path)
        if index<len(old_manifest['sources'][domain]) and digest!=old_manifest['sources'][domain][index]['sha256']:
            raise ValueError('Pinned original raw source changed')
        return domain,index,path,{'repository':repo,'revision':rev,'file':name,'sha256':digest}
    with cf.ThreadPoolExecutor(3) as pool:
        for domain,index,path,receipt in pool.map(fetch,[(d,i) for d in SOURCES for i in range(args.web_shards if d=='web' else args.shards)]):
            sources.setdefault(domain,[]).append((index,path,receipt))
    near=NearDuplicateIndex();found=set();seen=set(old_ids);rejected=Counter();train_counts={}
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    with cf.ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(tokenizer),old_ids)) as pool:
        for domain in SOURCES:
            paths=[path for _,path,_ in sources[domain]][:len(old_manifest['sources'][domain])]
            for batch in bounded_map(pool,rebuild_batch,interleaved_batches(domain,paths),args.workers*2):
                reserve(args.output)
                for identity,signature,sketch in batch:
                    if identity not in found:near.add(signature,sketch);found.add(identity)
                if domain_ids[domain]<=found:break
            if not domain_ids[domain]<=found:raise ValueError('Could not reconstruct every original '+domain+' document')
            print(json.dumps({'event':'original_index_rebuilt','domain':domain,'documents':len(found),'seconds':time.monotonic()-started}),flush=True)
        if found!=old_ids:raise AssertionError('Original duplicate index incomplete')
        db=sqlite3.connect(extension/'documents.sqlite')
        db.execute('create table documents(id text primary key,domain text,source_group text,split text,cohort text,offset integer,length integer,source text,license text,shard integer)')
        for domain,weight in [('web',.6),('math',.2),('code',.2)]:
            quota=int(args.additional_tokens*weight);accepted=offset=0;last_progress=-1
            with (extension/(domain+'.u16')).open('wb') as stream:
                for batch in bounded_map(pool,extension_batch,interleaved_batches(domain,[path for _,path,_ in sources[domain]]),args.workers*2):
                    reserve(args.output)
                    for doc in batch:
                        if 'reject' in doc:rejected[domain+'/'+doc['reject']]+=1;continue
                        if accepted>=quota:break
                        if doc['id'] in seen:rejected[domain+'/exact_duplicate']+=1;continue
                        reason=near.check(doc['signature'],doc['sketch'])
                        if reason:rejected[domain+'/'+reason]+=1;continue
                        seen.add(doc['id']);near.add(doc.pop('signature'),doc.pop('sketch'))
                        length=len(doc['tokens']);np.asarray(doc['tokens'],dtype=np.uint16).tofile(stream)
                        db.execute('insert into documents values(?,?,?,?,?,?,?,?,?,?)',
                            (doc['id'],domain,doc['group'],'train',None,offset,length,doc['source'],doc['license'],doc['shard']))
                        accepted+=length;offset+=length
                    db.commit()
                    if accepted//50_000_000!=last_progress:
                        last_progress=accepted//50_000_000
                        print(json.dumps({'event':'new_train','domain':domain,'accepted_targets':accepted,'seconds':time.monotonic()-started}),flush=True)
                    if accepted>=quota:break
            if accepted<quota:raise ValueError(f'Insufficient natural {domain} source: {accepted} < {quota}')
            train_counts[domain]=accepted
        db.execute('attach database ? as parent',(str(args.parent/'documents.sqlite'),))
        overlap=db.execute("select count(distinct n.source_group) from documents n join parent.documents p on n.source_group=p.source_group where p.split!='train'").fetchone()[0]
        if overlap:raise AssertionError('Appended TRAIN crosses original held-out source groups')
        extension_packing=freeze_packing(extension,db,train_counts,1024)
    packing=concatenate(args.parent,extension,args.output)
    for name in ['dev.json','test-sealed.json']:shutil.copy2(args.parent/name,args.output/name)
    report={'status':'PASS_FROZEN_NATURAL_EXTENSION','synthetic_generation':False,'parent_manifest_sha256':file_sha256(args.parent/'manifest.json'),
            'sources':{d:[receipt for _,_,receipt in values] for d,values in sources.items()},
            'original_duplicate_index_documents_reconstructed':len(found),'original_duplicate_index_complete':found==old_ids,
            'additional_domain_targets':train_counts,'source_group_overlap':overlap,'rejected':dict(rejected),
            'near_duplicate_audit':old_manifest['near_duplicate_audit'],'packing':packing,
            'dev_sha256':file_sha256(args.output/'dev.json'),'test_sha256':file_sha256(args.output/'test-sealed.json'),
            'original_dev_test_unchanged':all(file_sha256(args.parent/n)==file_sha256(args.output/n) for n in ['dev.json','test-sealed.json']),
            'extension_packing':extension_packing,'preparation_wall_seconds':time.monotonic()-started}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'real_targets':packing['real_targets'],'seconds':report['preparation_wall_seconds']}),flush=True)


if __name__=='__main__':main()
