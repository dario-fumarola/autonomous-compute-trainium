"""Bounded, source-separated reuse of admitted natural 32k-tokenizer documents."""
import argparse,json,shutil,sqlite3,time
from pathlib import Path
import numpy as np
from compute_trainium.training import file_sha256


def main(root):
    start=time.time();out=root/'probability-workspace-data-v1'
    if out.exists():raise FileExistsError('Preserve every preparation')
    if shutil.disk_usage(root).free<38*1024**3:raise RuntimeError('35GiB reserve plus complete data allowance')
    source=root/'natural-append-3250m/extension';db=source/'documents.sqlite'
    reserve=root/'residual-memory-attribution-v1/PRIVATE-reserved-sources.json'
    excluded={r['group'] for r in json.loads(reserve.read_text())}
    metadata=json.loads((source/'source.json').read_text())
    if metadata['schema']!='natural-learning-v1' or metadata['synthetic_generation']:
        raise AssertionError('Only accepted original natural source tokens')
    out.mkdir(mode=0o700)
    plan=dict(purpose='Source preparation, no model scoring or training',registered_unix=start,
        train_position_cap=640000000,context=512,tune_windows_per_domain=128,
        admitted_source='natural-append-3250m/extension',accepted_database_sha256=file_sha256(db),
        original_source_metadata_sha256=file_sha256(source/'source.json'),
        reserved_groups_sha256=file_sha256(reserve),tokenizer_revision=metadata['tokenizer_revision'],
        train_rule='Accepted TRAIN, source-group first8hex modulo10 nonzero, exclude320 reserved groups; id-sorted per domain, interleave at60/20/20 target ratio.',
        tune_rule='Accepted TRAIN source-group first8hex modulo10 zero; distinct groups, first eligible512-token window per group,128 per domain. Reused source population, not confirmation.',
        maximum_document_tokens=16384,original_targets_only=True,seconds_limit=900)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
    arrays={d:np.memmap(source/(d+'.u16'),dtype='<u2',mode='r') for d in ('web','math','code')}
    plan['raw_token_sha256']={d:file_sha256(source/(d+'.u16')) for d in arrays}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def stream(domain,split):
        rows=c.execute('select id,source_group,offset,length from documents where domain=? and split=? order by id',(domain,'train'))
        for id,group,offset,length in rows:
            if group in excluded or (int(group[:8],16)%10==0)!=(split=='tune'):continue
            if split=='tune' and length<513:continue
            n=min(length,plan['maximum_document_tokens'])
            ids=np.asarray(arrays[domain][offset:offset+n])
            if len(ids)!=n or ids.max()>=32000:raise AssertionError('Original token offset or vocabulary differs')
            yield id,group,ids
    # Independent connections keep three live ordered cursors independent.
    streams={d:stream(d,'train') for d in arrays}
    cap=plan['train_position_cap'];length=512
    tokens=np.lib.format.open_memmap(out/'train-tokens.npy',mode='w+',dtype=np.uint16,shape=(cap+length+1,))
    domains=np.lib.format.open_memmap(out/'train-domains.npy',mode='w+',dtype=np.uint8,shape=(cap+length+1,))
    counts={d:0 for d in arrays};groups=set();position=0;docs=0;weights={'web':.6,'math':.2,'code':.2}
    recipe=(out/'PRIVATE-training-documents.jsonl').open('x')
    try:
        while position<cap:
            domain=min(arrays,key=lambda d:(counts[d]/weights[d],d))
            id,group,ids=next(streams[domain]);n=min(len(ids),cap-position-2)
            if n<=0:break
            tokens[position]=1;domains[position]=0
            tokens[position+1:position+1+n]=ids[:n];domains[position+1:position+1+n]=1+list(arrays).index(domain)
            tokens[position+1+n]=2;domains[position+1+n]=0
            recipe.write(json.dumps(dict(id=id,group=group,domain=domain,start=position,source_tokens=n))+'\n')
            position+=n+2;counts[domain]+=n;groups.add(group);docs+=1
            if docs%1024==0:
                (out/'progress.json').write_text(json.dumps(dict(documents=docs,positions=position,seconds=time.time()-start)))
                if time.time()-start>900 or shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Source-preparation bound')
        windows=(position+length-2)//length;end=windows*length+1
        tokens[position:]=2;domains[position:]=0;tokens.flush();domains.flush();recipe.close()
        evaluation=[];evalgroups=set()
        for domain in arrays:
            used=0
            for id,group,ids in stream(domain,'tune'):
                if group in evalgroups:continue
                evaluation.append(dict(id=id,group=group,domain=domain,tokens=ids[:513].tolist()))
                used+=1;evalgroups.add(group)
                if used==128:break
            if used!=128:raise AssertionError('Frozen TUNE source yield shortfall')
        if groups&evalgroups or (groups|evalgroups)&excluded:raise AssertionError('Source-group overlap')
        np.save(out/'tune-tokens.npy',np.array([r['tokens'] for r in evaluation],dtype=np.uint16))
        (out/'PRIVATE-tune-sources.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='tokens'} for r in evaluation]))
        result=dict(status='PASS_PROBABILITY_WORKSPACE_NATURAL_SOURCES',context=512,windows=windows,valid_positions=end,
            train_target_counts=counts,train_documents=docs,train_source_groups=len(groups),tune_source_groups=len(evalgroups),
            source_group_overlap=0,reserved_group_overlap=0,model_scores=0,assigned_chips=0,optimizer_updates=0,
            files={f.name:file_sha256(f) for f in out.iterdir() if f.name.endswith(('.npy','.jsonl')) or f.name=='PRIVATE-tune-sources.json'},
            wall_seconds=time.time()-start,tokenizer_revision=metadata['tokenizer_revision'])
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:recipe.close();c.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
