"""Exact checks of source-authored arithmetic chains;no generated labels."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time

from compute_trainium.data import content_hash,source_group,split_for_group
from compute_trainium.source_arithmetic import original_chains,normalize
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='natural-source-chain-audit-v1'
PUBLIC=Path('results/natural_source_chain_audit_20261025')


def candidate(row):
    chain=next(original_chains(row['text']),None)
    if chain is None:return None
    text=row['text'];parts=[chain['expression'],*chain['intermediate_steps'],chain['conclusion']]
    expression=text[slice(*chain['expression'])]
    return dict(id=row['id'],group=row['group'],shard=row['shard'],original_utf8_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_spans=chain,utf8_spans=[[len(text[:a].encode()),len(text[:b].encode())] for a,b in parts],
        original_source_fragments=[text[a:b] for a,b in parts],
        exact_expression_group=hashlib.sha256(normalize(expression).encode()).hexdigest(),
        equality_checks_pass=True,training_view_created=False,generated_answer=False,generated_work=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();started=time.time();out=args.root/IDENTITY;out.mkdir()
    parent=args.root/'natural-worked-reasoning-audit-v1';old=json.loads((parent/'plan.json').read_text())
    selected=json.loads((parent/'selected-train-metadata.json').read_text())
    if file_sha256(parent/'selected-train-metadata.json')!=old['selected_metadata_sha256']:
        raise ValueError('Original accepted TRAIN identities changed')
    sources=['scripts/audit_natural_source_chains.py','compute_trainium/source_arithmetic.py',
             'tests/test_source_arithmetic.py','docs/NATURAL_SOURCE_CHAIN_AUDIT_20261025.md']
    plan=dict(identity=IDENTITY,registered_unix=started,seconds_limit=600,maximum_readers=4,
        source_sha256={p:file_sha256(p) for p in sources},parent_plan_sha256=file_sha256(parent/'plan.json'),
        selected_metadata_sha256=old['selected_metadata_sha256'],inputs=old['inputs'],
        original_sample_sha256=file_sha256(parent/'inspection-sample.jsonl'),
        scope=Path(sources[-1]).read_text(),research_optimizer_updates=0,assigned_chips=0,test_accessed=False)
    PUBLIC.mkdir(parents=True,exist_ok=False);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    result=dict(status='STOPPED_NATURAL_SOURCE_CHAIN_AUDIT');found=set();examples=[]
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if time.time()-started>600 or available<512*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Finite source-chain audit resource cap')
    def scan(item):
        import pyarrow.parquet as pq
        path=args.root/item['relative']
        if file_sha256(path)!=item['sha256']:raise ValueError('Pinned raw source changed')
        targets=defaultdict(dict)
        for record in selected:
            if record['shard']==item['shard']:targets[record['url']][record['id']]=record
        seen=set();records=[]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['url','text']):
            guard()
            for row in batch.to_pylist():
                eligible=targets.get(row['url'])
                if not eligible:continue
                key=content_hash(row['text'])
                if key not in eligible or key in seen:continue
                meta=eligible[key];group=source_group('math',row['url'])
                if split_for_group(group)!='train' or hashlib.sha256(group.encode()).hexdigest()!=meta['group']:
                    raise ValueError('Original TRAIN source group changed')
                seen.add(key);record=candidate({**meta,'text':row['text']})
                if record:records.append(record)
        return seen,records
    try:
        guard()
        with (parent/'inspection-sample.jsonl').open() as stream:sample=[json.loads(s) for s in stream]
        sample_candidates=[r for row in sample if (r:=candidate(row)) is not None]
        write_state(out/'sample-candidates.json',sample_candidates)
        with ThreadPoolExecutor(4) as pool:
            for future in as_completed([pool.submit(scan,item) for item in plan['inputs']]):
                identities,records=future.result()
                if found&identities:raise AssertionError('Duplicate recovered identity')
                found.update(identities);examples.extend(records)
        if found!={r['id'] for r in selected}:raise AssertionError('Incomplete accepted-source reconstruction')
        examples.sort(key=lambda r:r['id']);write_state(out/'source-span-candidates.json',examples)
        if any(file_sha256(p)!=s for p,s in plan['source_sha256'].items()):raise AssertionError('Frozen checker changed')
        result=dict(status='COMPLETE_NATURAL_SOURCE_CHAIN_FEASIBILITY_AUDIT',recovered_documents=len(found),
            matched_documents=len(examples),source_groups=len({r['group'] for r in examples}),
            distinct_exact_expression_groups=len({r['exact_expression_group'] for r in examples}),
            premise_operation_counts=dict(Counter(r['source_spans']['original_operations'] for r in examples)),
            inspection_sample_documents=len(sample),inspection_sample_matched_documents=len(sample_candidates),
            candidates_sha256=file_sha256(out/'source-span-candidates.json'),
            training_views_created=0,model_predictions_read=0,
            scope='Source-authored equality verification and span candidates only. Checker values never become answers or work. Sample is accepted TRAIN from six shards;not a population or learning result. No source text/fragments are published,no teacher/student model,generated research examples,held-out scoring or training admission.')
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(plan_sha256=file_sha256(out/'plan.json'),wall_seconds=time.time()-started,
            assigned_chips=0,research_optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        print(json.dumps({k:result[k] for k in ['status','wall_seconds']}),flush=True)


if __name__=='__main__':main()
