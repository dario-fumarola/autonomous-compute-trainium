"""Build a bounded index over admitted source bytes; no models or new tokens."""
import argparse,json,resource,shutil,sqlite3,time
from pathlib import Path
import numpy as np
from compute_trainium.training import file_sha256
from compute_trainium.indexed_natural_sources import RECORD_DTYPE,DOMAINS,IndexedNaturalSources

ID='indexed-natural-sources-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/indexed-sources-v1')

def main(root):
    start=time.time();out=root/ID
    if out.exists():raise FileExistsError('Preserve every source preparation')
    if shutil.disk_usage(root).free<37*1024**3:raise RuntimeError('2GiB index allowance plus protected reserve')
    parent=root/'natural-append-capacity-v4';source=parent/'extension';db=source/'documents.sqlite'
    audit=json.loads((parent/'independent-audit.json').read_text());meta=json.loads((source/'source.json').read_text())
    if audit['status']!='PASS_INDEPENDENT_CHAINED_EXTENSION_AUDIT' or audit['protected_group_overlap'] or meta['synthetic_generation'] or meta['tokenizer_revision']!='27d67f1b5f57dc0953326b2601d68371d40ea8da':raise AssertionError('Admitted original source and tokenizer required')
    reserves=[root/'residual-memory-attribution-v1/PRIVATE-reserved-sources.json',
      root/'probability-workspace-evaluation-v1/PRIVATE-broad-metadata.json',
      root/'probability-workspace-evaluation-v1/PRIVATE-dependency-metadata.json',
      root/'probability-structural-panel-v1/PRIVATE-metadata.json',
      root/'probability-fresh-broad-reserve-v2/PRIVATE-broad-metadata.json']
    excluded={r['group'] for path in reserves for r in json.loads(path.read_text())}
    out.mkdir(mode=0o700);PUBLIC.mkdir(parents=True)
    plan=dict(identity=ID,registered_unix=start,seconds_limit=1200,artifact_allowance_bytes=2*1024**3,
      source_relative=str(source.relative_to(root)),source_metadata_sha256=file_sha256(source/'source.json'),
      source_audit_sha256=file_sha256(parent/'independent-audit.json'),database_sha256=file_sha256(db),
      excluded_group_files={str(p.relative_to(root)):file_sha256(p) for p in reserves},
      requested_positions=6000000000,minimum_positions=5200000000,context=512,maximum_document_tokens=16384,
      selection='Accepted TRAIN; source-group first8hex modulo10 nonzero; all listed evaluation/reserve groups excluded; id order within each domain; deterministic60/20/20 real-token interleave.',
      capacity_rule='Before packing, use min(6B, floor(0.98 times smallest filtered per-domain token capacity divided by its mixture fraction)); stop below5.2B.',
      assigned_chips=0,optimizer_updates=0,model_scores=0,source_code_sha256=file_sha256(__file__))
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(PUBLIC/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if time.time()-start>1200 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>48*1024**3 or available<512*1024**3 or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Source preparation resource bound')
    connection=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
    weights=dict(zip(DOMAINS,(.6,.2,.2)))
    def eligible(domain):
        cursor=connection.execute('select id,source_group,offset,length from documents where domain=? and split=? order by id',(domain,'train'))
        for i,(id,group,offset,length) in enumerate(cursor):
            if i%8192==0:guard()
            if group in excluded or int(group[:8],16)%10==0 or length<1:continue
            yield id,group,offset,min(length,16384)
    try:
        capacity={d:sum(r[3] for r in eligible(d)) for d in DOMAINS}
        cap=min(6000000000,int(.98*min(capacity[d]/weights[d] for d in DOMAINS)))
        if cap<5200000000:raise RuntimeError('Filtered authentic source capacity below registered5.2B minimum')
        arrays={d:np.memmap(source/(d+'.u16'),dtype='<u2',mode='r') for d in DOMAINS}
        raw_hashes={d:file_sha256(source/(d+'.u16')) for d in DOMAINS}
        streams={d:eligible(d) for d in DOMAINS};counts={d:0 for d in DOMAINS};records=[];groups=set();position=0;documents=0
        with (out/'PRIVATE-training-documents.jsonl').open('x') as recipe:
            while position<cap-2:
                domain=min(DOMAINS,key=lambda d:(counts[d]/weights[d],d));id,group,offset,length=next(streams[domain]);n=min(length,cap-position-2)
                values=arrays[domain][offset:offset+n]
                if len(values)!=n or int(values.max())>=32000:raise AssertionError('Accepted source offset/vocabulary differs')
                records.append((position,offset,n,DOMAINS.index(domain)+1))
                recipe.write(json.dumps(dict(id=id,group=group,domain=domain,start=position,source_offset=offset,source_tokens=n))+'\n')
                counts[domain]+=n;groups.add(group);position+=n+2;documents+=1
                if documents%8192==0:
                    guard();(out/'progress.json').write_text(json.dumps(dict(documents=documents,positions=position,seconds=time.time()-start)))
        np.save(out/'PRIVATE-documents.npy',np.asarray(records,dtype=RECORD_DTYPE));del records
        windows=(position+510)//512
        report=dict(status='PASS_INDEXED_NATURAL_SOURCES',context=512,windows=windows,valid_positions=position,
          train_target_counts=counts,train_documents=documents,train_source_groups=len(groups),source_relative=str(source.relative_to(root)),
          source_group_overlap=len(groups&excluded),protected_group_overlap=len(groups&excluded),filtered_domain_capacity=capacity,
          accepted_position_cap=cap,raw_token_sha256=raw_hashes,tokenizer_revision=meta['tokenizer_revision'],
          assigned_chips=0,optimizer_updates=0,model_scores=0,original_source_bytes_copied=0,
          files={f.name:file_sha256(f) for f in out.iterdir() if f.name.endswith(('.npy','.jsonl'))})
        if report['source_group_overlap']:raise AssertionError('Reserved source group overlap')
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        # Independent random-range reconstruction against the source recipe,
        # including every virtual boundary and mask in selected windows.
        data=IndexedNaturalSources(out);rng=np.random.default_rng(9827)
        checks=sorted(set([0,1,windows-1,*rng.integers(0,windows,size=512).tolist()]))
        for index in checks:
            tokens,domains=data.window(index);left=index*512;right=left+513
            j=max(0,int(np.searchsorted(data.records['start'],left,side='right'))-1)
            reference=np.full(513,2,dtype=np.int64);mask=np.zeros(513,dtype=np.uint8)
            while j<len(data.records) and int(data.records[j]['start'])<right:
                row=data.records[j];startrow=int(row['start']);d=DOMAINS[int(row['domain'])-1]
                original=np.asarray(arrays[d][int(row['offset']):int(row['offset'])+int(row['length'])],dtype=np.int64)
                packed=np.concatenate(([1],original,[2]));dm=np.concatenate(([0],np.full(len(original),int(row['domain']),dtype=np.uint8),[0]))
                lo=max(left,startrow);hi=min(right,startrow+len(packed))
                reference[lo-left:hi-left]=packed[lo-startrow:hi-startrow];mask[lo-left:hi-left]=dm[lo-startrow:hi-startrow];j+=1
            if not np.array_equal(tokens,reference) or not np.array_equal(domains,mask) or data.target_count(index,1)!=int(np.count_nonzero(mask[1:])):raise AssertionError('Independent indexed source reconstruction failed')
        if data.target_count(0,windows)!=sum(counts.values()):raise AssertionError('Every original source label exactly once')
        report.update(independent_windows_reconstructed=len(checks),all_real_targets_counted_once=True,
          wall_seconds=time.time()-start,maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          index_artifact_bytes=sum(f.stat().st_size for f in out.iterdir() if f.is_file()))
        if report['index_artifact_bytes']>2*1024**3:raise AssertionError('Registered artifact allowance exceeded')
        guard();(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(PUBLIC/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    except BaseException as error:
        failure=dict(status='STOPPED_INDEXED_SOURCE_PREPARATION',error_type=type(error).__name__,error=str(error),wall_seconds=time.time()-start,assigned_chips=0,optimizer_updates=0,model_scores=0)
        (out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n');(PUBLIC/'failure.json').write_text(json.dumps(failure,indent=2)+'\n');raise
    finally:connection.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);p=a.parse_args();main(p.root)
