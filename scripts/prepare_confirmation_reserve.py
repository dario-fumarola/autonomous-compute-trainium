"""Bounded, memory-limited fresh natural held-out groups; no model scoring.

Accepted-ID arrays are mapped read-only. Prior raw sketches stream against a
small frozen candidate index, preserving the existing near-duplicate rule.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sqlite3
import subprocess
import time
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from compute_trainium.corpus_audit import NearDuplicateIndex,bottom_k_jaccard,text_sketch
from compute_trainium.data import content_hash,source_group,split_for_group
from compute_trainium.heldout_screen import CandidateIndex,PriorNeighborhoods,locate_hashes,panel_role
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.prepare_chained_extension import lineage,old_raw_paths,source_key
from scripts.prepare_learning_corpus import bounded_map,process_batch,reserve,worker_init
from scripts.prepare_natural import rows


DOMAINS=['web','math','code'];ROLES=['tune','curve','test']


def mapped_init(directory):
    global HASHES
    HASHES=[np.load(p,mmap_mode='r') for p in sorted((Path(directory)/'index').glob('ids-*.npy'))]


def collect_init(directory,tokenizer,groups):
    global PROTECTED
    mapped_init(directory);worker_init(tokenizer);PROTECTED=set(groups)


def indexed_batches(domain,paths):
    import itertools
    readers=[(i,iter(rows(path)),0) for i,path in enumerate(paths)]
    while readers:
        following=[]
        for shard,reader,start in readers:
            values=list(itertools.islice(reader,256))
            if values:yield domain,shard,start,values;following.append((shard,reader,start+len(values)))
        readers=following


def candidates(batch):
    domain,shard,start,values=batch;possible=[]
    for index,row in enumerate(values):
        text=row.get('text',row.get('content',''))
        if not text or len(text)>100000:continue
        try:group=source_group(domain,row.get('repo_name','') if domain=='code' else row.get('url',''))
        except ValueError:continue
        group_hash=hashlib.sha256(group.encode()).hexdigest()
        if split_for_group(group)=='train' or group_hash in PROTECTED:continue
        identity=content_hash(text)
        if int(identity[:8],16)%8:continue
        possible.append((index,row,identity,group_hash))
    generation,_=locate_hashes(HASHES,[x[2] for x in possible]);result=[]
    for old,(index,row,identity,group_hash) in zip(generation,possible):
        if old>=0:continue
        for doc in process_batch([(domain,shard,row)]):
            if 'reject' in doc:continue
            if doc['id']!=identity or doc['group']!=group_hash:raise AssertionError('Tokenizer/provenance identity mismatch')
            doc.update(role=panel_role(group_hash),raw_row=start+index);result.append(doc)
    return result


def scan_init(directory):
    global INDEX,FOUND,REJECTED
    root=Path(directory);mapped_init(directory)
    sig=np.load(root/'pool-signatures.npy',mmap_mode='r');sketch=np.load(root/'pool-sketches.npy',mmap_mode='r')
    length=np.load(root/'pool-sketch-lengths.npy',mmap_mode='r')
    INDEX=CandidateIndex(sig,[sketch[i,:int(n)] for i,n in enumerate(length)])
    FOUND=[np.load(p,mmap_mode='r+') for p in sorted((root/'index').glob('found-*.npy'))]
    REJECTED=np.load(root/'pool-rejected.npy',mmap_mode='r')


def scan_old(batch):
    domain,shard,start,values=batch;eligible=[]
    for row in values:
        text=row.get('text',row.get('content',''))
        if text and len(text)<=100000:eligible.append((content_hash(text),text))
    generation,position=locate_hashes(HASHES,[v[0] for v in eligible]);hits=[];accepted=0
    for gen,where,(_,text) in zip(generation,position,eligible):
        if gen<0:continue
        signature,sketch=text_sketch(text)
        for candidate in INDEX.matching(signature):
            if not REJECTED[candidate]:hits.append((candidate,int(gen),int(where),bottom_k_jaccard(sketch,INDEX.sketches[candidate])>=.8))
        FOUND[gen][where]=True;accepted+=1
    return {'raw_rows':len(values),'accepted_occurrences':accepted,'hits':hits}


def write_index(directory,number,values,count):
    hashes=np.lib.format.open_memmap(directory/f'ids-{number:02}.npy',mode='w+',dtype='S64',shape=(count,))
    domains=np.lib.format.open_memmap(directory/f'domains-{number:02}.npy',mode='w+',dtype=np.uint8,shape=(count,))
    last=None;index=0
    for identity,domain in values:
        if len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity) or (last is not None and identity<=last):
            raise ValueError('Accepted identities are not unique ordered SHA256 values')
        hashes[index]=identity;domains[index]=DOMAINS.index(domain)+1;last=identity;index+=1
    if index!=count:raise AssertionError('Accepted ID count changed during export')
    hashes.flush();domains.flush()
    found=np.lib.format.open_memmap(directory/f'found-{number:02}.npy',mode='w+',dtype=bool,shape=(count,));found[:]=False;found.flush()


def run(a):
    started=time.time();reserve(a.output);manifests,databases=lineage(a.generations)
    container_pid=subprocess.check_output(['docker','inspect','--format','{{.State.Pid}}','autonomous-compute-neuron233'],text=True).strip()
    cgroup=Path('/proc/'+container_pid+'/cgroup').read_text().strip().split('::',1)[1]
    memory=Path('/sys/fs/cgroup')/cgroup.lstrip('/')
    limit=int((memory/'memory.max').read_text())
    def margin():
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        return available-512*1024**3-max(0,limit-int((memory/'memory.current').read_text()))
    minimum_margin=margin()
    if minimum_margin<(8+1.5*a.workers)*1024**3:raise RuntimeError('Insufficient conservative CPU-work memory allowance')
    def guard():
        nonlocal minimum_margin
        reserve(a.output)
        if time.time()>a.deadline_unix:raise TimeoutError('Fresh data reserve deadline')
        latest=margin();minimum_margin=min(minimum_margin,latest)
        if latest<4*1024**3:raise RuntimeError('Fresh data preparation must yield the native memory allowance')
    tokenizer=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    protected=[a.heldout/'dev.json',a.heldout/'test-sealed.json']
    source_files=['scripts/prepare_confirmation_reserve.py','compute_trainium/heldout_screen.py',
        'scripts/prepare_chained_extension.py','scripts/prepare_learning_corpus.py','scripts/prepare_natural.py',
        'compute_trainium/data.py','compute_trainium/corpus_audit.py']
    plan={'identity':'fresh-confirmation-reserve-v1','purpose':'DATA_ONLY_NO_MODEL_SELECTION_OR_SCORING',
        'registered_unix':time.time(),'deadline_unix':a.deadline_unix,'workers':a.workers,'synthetic_generation':False,
        'generation_manifest_sha256':[file_sha256(g/'manifest.json') for g in a.generations],
        'prior_database_sha256':[file_sha256(p) for p in databases],'protected_panel_sha256':[file_sha256(p) for p in protected],
        'tokenizer_sha256':file_sha256(tokenizer),'source_sha256':{f:file_sha256(f) for f in source_files},
        'selection':'Unused global natural-v1 heldout groups, exclude all prior accepted IDs and protected groups; content hash mod8=0; confirmation-reserve-v1 group hash assigns tune/curve/test; at most2 documents/group/domain/role. Keep up to1200 candidates/domain/role in raw receipt/interleaved-row order, then admit in content-hash order.',
        'real_target_quota_per_domain_role':250000,'minimum_groups_per_domain_role':100,
        'near_duplicate_rule':'Same32 minhash/8x4 bands, bottom256 Jaccard0.8 and distinct-prior-neighbor cutoff256. Inverted small candidate index, mapped accepted IDs and boolean reconstruction coverage.',
        'terminal_attempt_registered':False,'test_scored':False}
    write_state(a.output/'plan.json',plan);guard();index=a.output/'index';index.mkdir();groups=set()
    for number,path in enumerate(databases):
        db=sqlite3.connect(f'file:{path}?mode=ro',uri=True);count=db.execute('select count(*) from documents').fetchone()[0]
        write_index(index,number,db.execute('select id,domain from documents order by id'),count)
        groups.update(x[0] for x in db.execute("select distinct source_group from documents where split!='train'"));db.close();guard()
    arrays=[np.load(p,mmap_mode='r') for p in sorted(index.glob('ids-*.npy'))];extra={}
    for path in protected:
        for doc in json.loads(path.read_text()):
            groups.add(doc['group']);extra[doc['id']]=doc['domain']
    keys=sorted(extra);old,_=locate_hashes(arrays,keys)
    extra_rows=[(key,extra[key]) for key,gen in zip(keys,old) if gen<0]
    write_index(index,len(databases),extra_rows,len(extra_rows));guard()
    raw=old_raw_paths(a.generations,manifests,tokenizer);paths={};receipts={}
    for domain in DOMAINS:
        ordered=sorted((key,value) for key,value in raw.items() if key[0]==domain)
        paths[domain]=[value[0] for _,value in ordered];receipts[domain]=[value[1] for _,value in ordered]
    write_state(a.output/'raw-source-map.json',[{'domain':domain,'path':str(path),'receipt':receipt}
        for domain in DOMAINS for path,receipt in zip(paths[domain],receipts[domain])])
    print(json.dumps({'event':'mapped_prior_indices_ready','old_documents':sum(len(x) for x in arrays)+len(extra_rows),
        'protected_groups':len(groups),'seconds':time.time()-started}),flush=True)
    pools=defaultdict(list);group_counts=Counter();seen=set()
    with cf.ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=collect_init,
            initargs=(str(a.output),str(tokenizer),groups)) as executor:
        for domain in DOMAINS:
            for batch in bounded_map(executor,candidates,indexed_batches(domain,paths[domain]),a.workers*2):
                guard()
                for doc in batch:
                    key=(domain,doc['role']);group_key=(*key,doc['group'])
                    if len(pools[key])>=1200 or group_counts[group_key]>=2 or doc['id'] in seen:continue
                    pools[key].append(doc);group_counts[group_key]+=1;seen.add(doc['id'])
                if all(len(pools[domain,role])>=1200 for role in ROLES):break
            if any(sum(len(d['tokens']) for d in pools[domain,role])<250000 or len({d['group'] for d in pools[domain,role]})<100 for role in ROLES):
                raise ValueError('Insufficient raw fresh candidate pool: '+domain)
    pool=sorted([d for values in pools.values() for d in values],key=lambda d:d['id']);del pools
    signatures=np.stack([d.pop('signature') for d in pool]);sketches=[d.pop('sketch') for d in pool]
    np.save(a.output/'pool-signatures.npy',signatures)
    padded=np.full((len(pool),256),np.iinfo(np.uint64).max,dtype=np.uint64)
    for i,sketch in enumerate(sketches):padded[i,:len(sketch)]=sketch
    np.save(a.output/'pool-sketches.npy',padded);np.save(a.output/'pool-sketch-lengths.npy',np.asarray(list(map(len,sketches)),dtype=np.int16))
    write_state(a.output/'private-candidate-pool.json',pool)
    rejected=np.lib.format.open_memmap(a.output/'pool-rejected.npy',mode='w+',dtype=bool,shape=(len(pool),));rejected[:]=False;rejected.flush()
    prior=PriorNeighborhoods(len(pool));found=[np.load(p,mmap_mode='r') for p in sorted(index.glob('found-*.npy'))]
    labels=[np.load(p,mmap_mode='r') for p in sorted(index.glob('domains-*.npy'))]
    print(json.dumps({'event':'candidate_pool_frozen','candidates':len(pool),'pool_sha256':file_sha256(a.output/'private-candidate-pool.json'),'seconds':time.time()-started}),flush=True)
    with cf.ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=scan_init,initargs=(str(a.output),)) as executor:
        for domain in DOMAINS:
            for iteration,batch in enumerate(bounded_map(executor,scan_old,indexed_batches(domain,paths[domain]),a.workers*2)):
                guard()
                for candidate,gen,where,near in batch['hits']:
                    prior.observe(candidate,(gen,where),near)
                    if prior.reason(candidate):rejected[candidate]=True
                # Consume every submitted result before trusting shared coverage.
                # A worker may set its bitmap before its neighbor hits reach us.
            if not all(np.all(f[d==DOMAINS.index(domain)+1]) for f,d in zip(found,labels)):
                raise ValueError('Prior raw near-duplicate reconstruction incomplete: '+domain)
            print(json.dumps({'event':'prior_reconstruction_complete','domain':domain,'found':sum(int(f.sum()) for f in found),
                'rejected_candidates':int(rejected.sum()),'seconds':time.time()-started}),flush=True)
    if not all(np.all(f) for f in found):raise AssertionError('Not every prior ID was screened')
    near=NearDuplicateIndex();chosen=[];counts={};reasons=Counter();accepted_groups={};selected_group_counts=Counter()
    for domain in DOMAINS:
        for role in ROLES:
            total=0;role_groups=set();docs=0
            for candidate,doc in enumerate(pool):
                if doc['domain']!=domain or doc['role']!=role:continue
                additional=set()
                for key in near.keys(signatures[candidate]):additional.update(near.bands.get(key,()))
                reason=prior.reason(candidate,len(additional)) or near.check(signatures[candidate],sketches[candidate])
                if reason:reasons[reason]+=1;continue
                if doc['group'] in accepted_groups and accepted_groups[doc['group']]!=role:raise AssertionError('Cross-role source group')
                if selected_group_counts[domain,role,doc['group']]>=2:continue
                near.add(signatures[candidate],sketches[candidate]);accepted_groups[doc['group']]=role
                selected_group_counts[domain,role,doc['group']]+=1;role_groups.add(doc['group']);total+=len(doc['tokens']);docs+=1
                chosen.append(dict(doc,split='test' if role=='test' else 'dev',cohort=None if role=='test' else role))
                if total>=250000 and len(role_groups)>=100:break
            if total<250000 or len(role_groups)<100:raise ValueError('Insufficient screened fresh '+domain+'/'+role)
            counts[domain+'/'+role]={'real_targets':total,'documents':docs,'source_groups':len(role_groups)}
    guard()
    for split,name in [('dev','dev.json'),('test','test-sealed.json')]:
        write_state(a.output/name,[d for d in chosen if d['split']==split]);(a.output/name).chmod(0o600)
    if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Frozen data preparation source changed')
    if [file_sha256(p) for p in databases]!=plan['prior_database_sha256']:raise AssertionError('Accepted metadata changed')
    report={'status':'PASS_FRESH_CONFIRMATION_RESERVE_PREPARATION_AUDIT_PENDING','plan_sha256':file_sha256(a.output/'plan.json'),
        'synthetic_generation':False,'counts':counts,'sources':receipts,'old_documents_reconstructed':sum(len(f) for f in found),
        'all_prior_raw_sketches_checked':True,'candidate_count':len(pool),'rejected':dict(reasons),
        'dev_sha256':file_sha256(a.output/'dev.json'),'test_sha256':file_sha256(a.output/'test-sealed.json'),
        'source_group_overlap_with_prior':0,'cross_role_source_group_overlap':0,'preparation_wall_seconds':time.time()-started,
        'minimum_sampled_conservative_memory_margin_bytes':minimum_margin,
        'cpu_seconds':{'parent_user':resource.getrusage(resource.RUSAGE_SELF).ru_utime,
            'parent_system':resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            'reaped_workers_user':resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
            'reaped_workers_system':resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime},
        'scope':'Natural-data-only fresh source-group reserve with the existing approximate near-duplicate rule and every-prior-ID coverage. Small candidate index replaces a corpus-sized Python sketch index. Independent SQL/provenance/raw-token audit remains required. No model/exit/LR selection, terminal registration, alpha spending or TEST scoring.'}
    write_state(a.output/'manifest.json',report);print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--generations',nargs='+',type=Path,required=True)
    p.add_argument('--heldout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args()
    if not 1<=a.workers<=12 or not time.time()+600<a.deadline_unix<=time.time()+7200:raise ValueError('Bounded8-worker/2-hour preparation required')
    if Path.cwd().resolve() in a.output.resolve().parents:raise ValueError('Private data must remain outside Git')
    a.output.mkdir(mode=0o700,exist_ok=False);started=time.time();os.nice(10)
    try:run(a)
    except BaseException as error:
        write_state(a.output/'failure.json',{'status':'STOPPED_FRESH_RESERVE_PREPARATION','error':repr(error),'wall_seconds':time.time()-started});raise


if __name__=='__main__':main()
