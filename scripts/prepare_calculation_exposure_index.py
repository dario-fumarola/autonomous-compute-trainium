"""Prospective uniform-label source selection; no model or scientific fit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_latent_exposure_sources import guard,read,endpoint

ID='calculation-exposure-index-v1'
PUBLIC=Path('results/calculation_document_pool_20261030/index-v1')


def main(root, *, recovery=False):
    import math
    import mmap
    import numpy as np
    from compute_trainium.indexed_natural_sources import IndexedNaturalSources,RECORD_DTYPE,DOMAINS
    from compute_trainium.interleaved_workspace import matrix_work
    from compute_trainium.model import ModelConfig
    from compute_trainium.calculation_exposure import intact_exposure
    start=time.time();identity=ID+('-recovery-v1' if recovery else '')
    public=Path('results/calculation_document_pool_20261030/index-recovery-v1') if recovery else PUBLIC
    out=root/identity;parent=root/'indexed-natural-sources-v1';poolroot=root/'calculation-document-pool-v1'
    if out.exists() or public.exists():raise FileExistsError('Preserve every preparation')
    ir=read(parent/'report.json');ip=read(parent/'plan.json');pr=read(poolroot/'report.json');pp=read(poolroot/'plan.json')
    if pr['status']!='COMPLETE_AUTHENTIC_CALCULATION_DOCUMENT_POOL':raise AssertionError('Verified original pool required')
    files=('scripts/prepare_calculation_exposure_index.py','scripts/audit_latent_exposure_sources.py',
        'compute_trainium/indexed_natural_sources.py','compute_trainium/interleaved_workspace.py','compute_trainium/model.py',
        'compute_trainium/calculation_exposure.py','tests/test_calculation_exposure.py',
        'docs/CALCULATION_EXPOSURE_INDEX_20261030.md')
    plan=dict(identity=identity,registered_unix=start,absolute_deadline_unix=start+600,
        requested_virtual_positions=487000000,context=512,artifact_allowance_bytes=256*1024**2,
        source_sha256={f:file_sha256(f) for f in files},source_relative=ir['source_relative'],
        parent_report_sha256=file_sha256(parent/'report.json'),pool_report_sha256=file_sha256(poolroot/'report.json'),
        pool_sha256=pr['files']['PRIVATE-pool.json'],recipe_sha256=ir['files']['PRIVATE-training-documents.jsonl'],
        maximum_original_position=pp['maximum_virtual_position'],excluded_group_files=ip['excluded_group_files'],
        assigned_chips=0,optimizer_updates=0,model_scores=0,scientific_fit_admitted=False,
        scope=Path('docs/CALCULATION_EXPOSURE_INDEX_20261030.md').read_text())
    if recovery:
        old=root/ID;failure=read(old/'report.json');prior=read(old/'plan.json')
        if failure['status']!='STOPPED_CALCULATION_EXPOSURE_INDEX' or failure['peak_rss_bytes']<=8*1024**3:
            raise AssertionError('Original memory-bound stop required')
        plan.update(absolute_deadline_unix=prior['absolute_deadline_unix'],
            recovery_of=dict(plan_sha256=file_sha256(old/'plan.json'),report_sha256=file_sha256(old/'report.json'),
                partial_recipe_sha256=file_sha256(old/'PRIVATE-training-documents.jsonl')))
        extra='docs/CALCULATION_INDEX_MEMORY_RECOVERY_20261030.md'
        plan['source_sha256'][extra]=file_sha256(extra)
    guard(root,plan);out.mkdir(mode=0o700);public.mkdir(parents=True)
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    result=dict(status='STOPPED_CALCULATION_EXPOSURE_INDEX')
    try:
        if (file_sha256(poolroot/'PRIVATE-pool.json')!=plan['pool_sha256'] or
            file_sha256(parent/'PRIVATE-training-documents.jsonl')!=plan['recipe_sha256']):
            raise AssertionError('Frozen source indices changed')
        for domain,h in ir['raw_token_sha256'].items():
            if file_sha256(root/ir['source_relative']/(domain+'.u16'))!=h:raise AssertionError('Original source bytes changed')
        pool=read(poolroot/'PRIVATE-pool.json');byid={r['id']:r for r in pool}
        excluded=set()
        for f,h in plan['excluded_group_files'].items():
            if file_sha256(root/f)!=h:raise AssertionError('Protected registry changed')
            excluded.update(r['group'] for r in read(root/f))
        def original(kind):
            with (parent/'PRIVATE-training-documents.jsonl').open() as stream:
                for i,line in enumerate(stream):
                    row=json.loads(line)
                    if row['start']>=plan['maximum_original_position']:break
                    if i%20000==0:guard(root,plan)
                    if row['domain']==kind and row['id'] not in byid:
                        length=min(row['source_tokens'],plan['maximum_original_position']-row['start']-1)
                        if length>0:yield {**row,'source_tokens':length}
        streams={'web':original('web'),'math-other':original('math'),'code':original('code'),'math-calculation':iter(pool)}
        weights={'web':.6,'math-other':.1,'math-calculation':.1,'code':.2}
        counts=Counter();domain_counts=Counter();groups=set();seen=set();records=[];calculations=[];position=0
        arrays={d:np.memmap(root/ir['source_relative']/(d+'.u16'),dtype='<u2',mode='r') for d in DOMAINS}
        for array in arrays.values():array._mmap.madvise(mmap.MADV_RANDOM)
        with (out/'PRIVATE-training-documents.jsonl').open('x') as recipe:
            while position<plan['requested_virtual_positions']-2:
                stream=min(weights,key=lambda d:(counts[d]/weights[d],d))
                row=next(streams[stream]);domain=row['domain'];n=min(row['source_tokens'],plan['requested_virtual_positions']-position-2)
                if row['id'] in seen or row['group'] in excluded or int(row['group'][:8],16)%10==0:
                    raise AssertionError('Duplicate document or protected source overlap')
                seen.add(row['id']);groups.add(row['group'])
                values=arrays[domain][row['source_offset']:row['source_offset']+n]
                if len(values)!=n or int(values.max())>=32000:raise AssertionError('Source range/vocabulary mismatch')
                records.append((position,row['source_offset'],n,DOMAINS.index(domain)+1))
                recipe.write(json.dumps(dict(id=row['id'],group=row['group'],domain=domain,start=position,
                    source_offset=row['source_offset'],source_tokens=n,selection_stream=stream))+'\n')
                if stream=='math-calculation':
                    for span in row['calculation_spans']:
                        if span['target_tokens'][1]<=n:calculations.append(dict(start=position,**span))
                counts[stream]+=n;domain_counts[domain]+=n;position+=n+2
                if len(records)%512==0:
                    for array in arrays.values():array._mmap.madvise(mmap.MADV_DONTNEED)
                    guard(root,plan)
        if recovery:
            with (root/ID/'PRIVATE-training-documents.jsonl').open('rb') as old, (out/'PRIVATE-training-documents.jsonl').open('rb') as new:
                for line in old:
                    if new.readline()!=line:raise AssertionError('Recovered source selection differs from original prefix')
        np.save(out/'PRIVATE-documents.npy',np.asarray(records,dtype=RECORD_DTYPE));del records
        windows=(position+510)//512
        accepted=dict(status='PASS_INDEXED_NATURAL_SOURCES',context=512,windows=windows,valid_positions=position,
            source_relative=ir['source_relative'],train_target_counts=dict(domain_counts),train_documents=len(seen),
            train_source_groups=len(groups),protected_group_overlap=0,source_group_overlap=0,
            raw_token_sha256=ir['raw_token_sha256'],tokenizer_revision=ir['tokenizer_revision'],
            files={f.name:file_sha256(f) for f in out.iterdir() if f.suffix in ('.npy','.jsonl')})
        write_state(out/'report.json',accepted);data=IndexedNaturalSources(out)
        if data.target_count(0,windows)!=sum(counts.values()):raise AssertionError('All authentic targets exactly once')
        checks=sorted(set([0,1,windows-1,*np.random.default_rng(84019).integers(0,windows,size=512).tolist()]))
        for index in checks:
            token,domain=data.window(index);left=index*512;right=left+513
            j=max(0,int(np.searchsorted(data.starts,left,side='right'))-1)
            expected=np.full(513,2,dtype=np.int64);mask=np.zeros(513,dtype=np.uint8)
            while j<len(data.records) and int(data.records[j]['start'])<right:
                row=data.records[j];s=int(row['start']);d=int(row['domain']);length=int(row['length']);offset=int(row['offset'])
                raw=np.asarray(arrays[DOMAINS[d-1]][offset:offset+length],dtype=np.int64)
                packed=np.concatenate(([1],raw,[2]));labels=np.concatenate(([0],np.full(length,d,dtype=np.uint8),[0]))
                lo=max(left,s);hi=min(right,s+length+2)
                expected[lo-left:hi-left]=packed[lo-s:hi-s];mask[lo-left:hi-left]=labels[lo-s:hi-s];j+=1
            if not np.array_equal(token,expected) or not np.array_equal(domain,mask) or data.target_count(index,1)!=np.count_nonzero(mask[1:]):
                raise AssertionError('Independent original-window reconstruction')
            for array in [*arrays.values(),*data.arrays.values()]:array._mmap.madvise(mmap.MADV_DONTNEED)
        latent=read(root/'interleaved-mixed-round-tuning-v1/plan.json');ordinary=read(root/'interleaved-tied-ordinary-tuning-v1/plan.json')
        u=endpoint(data,128000000)
        lc=ModelConfig(**latent['model_config']);oc=ModelConfig(**ordinary['model_config'])
        work=sum(((u+1)//2 if k==2 else u//2)*64*matrix_work(lc,512,k,mode='latent',batch=2)['training_matmul_flops'] for k in (2,3))
        matched=math.ceil(work/(64*matrix_work(oc,512,0,mode='ordinary',batch=2)['training_matmul_flops']))
        exposure={}
        for name,steps in [('equal_data_128M',u),('nominal_matched_work',matched)]:
            end=steps*128*512+1
            exposure[name]=dict(updates=steps,real_targets=data.target_count(0,steps*128),
                **intact_exposure(calculations,end))
        size=sum(f.stat().st_size for f in out.iterdir() if f.is_file())
        if size>plan['artifact_allowance_bytes']:raise RuntimeError('Index artifact allowance')
        guard(root,plan)
        result.update(**accepted,selection_stream_tokens=dict(counts),independent_windows_reconstructed=len(checks),
            all_real_targets_counted_once=True,original_source_bytes_copied=0,within_index_document_repeats=0,
            prior_research_source_reuse=True,prospective_exposure=exposure,index_artifact_bytes=size,
            original_partial_recipe_prefix_identical=True if recovery else None)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            assigned_chips=0,optimizer_updates=0,model_scores=0,scientific_fit_admitted=False,
            terminal_alpha_spent=0,plan_sha256=file_sha256(out/'plan.json'))
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--recovery',action='store_true')
    a=p.parse_args();main(a.root,recovery=a.recovery)
