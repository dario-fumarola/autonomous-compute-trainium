"""Recover accepted TRAIN sources and sample lexical candidates for inspection."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time

from compute_trainium.data import content_hash,source_group,split_for_group
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='natural-worked-reasoning-audit-v1'
PUBLIC=Path('results/natural_worked_reasoning_audit_20261025')
PROBLEM=re.compile(r'(?im)^\s*(?:#+\s*)?(?:problem|question|example|exercise)\b')
SOLUTION=re.compile(r'(?im)^\s*(?:#+\s*)?(?:solution|proof|working|answer)\b')
CONCLUSION=re.compile(r'(?i)\b(?:therefore|hence|thus|final answer)\b')
MATH=re.compile(r'(?:[=+*/]|\\(?:frac|sqrt|times|cdot))')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();started=time.time();out=args.root/IDENTITY
    out.mkdir(exist_ok=False);PUBLIC.mkdir(parents=True,exist_ok=False)
    corpus=args.root/'natural-append-6500m-v2';database=corpus/'extension/documents.sqlite'
    connection=sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True)
    rows=connection.execute("select id,source_group,source,shard,length from documents where domain='math' and split='train' and shard>=18 order by id limit 20000").fetchall();connection.close()
    if len(rows)!=20000:raise RuntimeError('Prespecified accepted TRAIN sample unavailable')
    selected=[dict(id=i,group=g,url=u,shard=s,source_tokens=n) for i,g,u,s,n in rows]
    write_state(out/'selected-train-metadata.json',selected)
    parent=json.loads((corpus/'manifest.json').read_text());inputs=[]
    for shard in range(18,24):
        file=corpus/'raw'/f'math-{shard}.parquet';expected=parent['sources']['math'][shard]['sha256']
        if file_sha256(file)!=expected:raise AssertionError('Pinned raw source changed')
        inputs.append(dict(shard=shard,relative=str(file.relative_to(args.root)),sha256=expected))
    source_files=['scripts/audit_natural_worked_reasoning.py','docs/NATURAL_WORKED_REASONING_AUDIT_20261025.md','compute_trainium/data.py']
    plan=dict(identity=IDENTITY,registered_unix=started,seconds_limit=3600,maximum_readers=4,
        selected_train_documents=20000,selection='Accepted TRAIN shards18..23,content-ID ascending',
        source_sha256={p:file_sha256(p) for p in source_files},inputs=inputs,
        selected_metadata_sha256=file_sha256(out/'selected-train-metadata.json'),
        parent_manifest_sha256=file_sha256(corpus/'manifest.json'),database_sha256=file_sha256(database),
        candidate_classes=['explicit_problem_solution','math_conclusion'],max_per_class=100,max_per_group=5,
        research_optimizer_updates=0,assigned_chips=0,test_accessed=False,scope=Path(source_files[1]).read_text())
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    result=dict(status='STOPPED_NATURAL_WORKED_REASONING_AUDIT');recovered={};candidates=[];counts=Counter()
    def guard():
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if time.time()-started>3600 or available<512*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Finite CPU data-audit resource cap')
    def scan(item):
        import pyarrow.parquet as pq
        guard();targets=defaultdict(dict)
        for record in selected:
            if record['shard']==item['shard']:targets[record['url']][record['id']]=record
        found={};local=[];stats=Counter()
        for batch in pq.ParquetFile(args.root/item['relative']).iter_batches(batch_size=256,columns=['url','text']):
            guard()
            for row in batch.to_pylist():
                eligible=targets.get(row['url'])
                if not eligible:continue
                identity=content_hash(row['text'])
                if identity not in eligible or identity in found:continue
                record=eligible[identity];group=source_group('math',row['url'])
                if split_for_group(group)!='train' or hashlib.sha256(group.encode()).hexdigest()!=record['group']:
                    raise AssertionError('Source group is not the accepted TRAIN identity')
                found[identity]=True;stats['recovered']+=1;text=row['text']
                problem=PROBLEM.search(text);solution=SOLUTION.search(text);conclusion=CONCLUSION.search(text)
                math_material=len(re.findall(r'\d',text))>=3 and len(MATH.findall(text))>=2
                kind=('explicit_problem_solution' if problem and solution and problem.start()<solution.start() and math_material
                      else 'math_conclusion' if conclusion and math_material else None)
                if kind:
                    stats[kind]+=1
                    local.append(dict(**record,candidate_class=kind,text=text,
                        cues={k:([m.start(),m.end()] if m else None) for k,m in [('problem',problem),('solution',solution),('conclusion',conclusion)]},
                        review_status='unreviewed',premise_sufficient=None,original_conclusion_identified=None))
        return found,local,stats
    try:
        guard()
        with ThreadPoolExecutor(4) as pool:
            for future in as_completed([pool.submit(scan,item) for item in inputs]):
                found,local,stats=future.result()
                if set(found)&set(recovered):raise AssertionError('An accepted identity recovered twice')
                recovered.update(found);candidates.extend(local);counts.update(stats)
                write_state(out/'progress.json',dict(recovered=len(recovered),candidate_counts=dict(counts),wall_seconds=time.time()-started))
        if set(recovered)!={r['id'] for r in selected}:raise AssertionError('Not every selected accepted source was recovered')
        sample=[];group_counts=Counter();class_counts=Counter()
        for record in sorted(candidates,key=lambda r:hashlib.sha256(('worked-audit-v1:'+r['id']).encode()).hexdigest()):
            kind=record['candidate_class'];group=record['group']
            if class_counts[kind]>=100 or group_counts[group]>=5:continue
            sample.append(record);class_counts[kind]+=1;group_counts[group]+=1
        private=out/'inspection-sample.jsonl'
        with private.open('x') as stream:
            for record in sample:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        guard()
        if any(file_sha256(p)!=s for p,s in plan['source_sha256'].items()):raise AssertionError('Frozen audit source changed')
        result=dict(status='COMPLETE_TRAIN_ONLY_WORKED_REASONING_CANDIDATE_INVENTORY',
            selected_documents=len(selected),recovered_documents=len(recovered),candidate_counts=dict(counts),
            sampled_candidates=len(sample),sampled_by_class=dict(class_counts),sampled_source_groups=len(group_counts),
            inspection_sample_sha256=file_sha256(private),manual_reviews_completed=0,
            training_views_created=0,scientific_training_admitted=False,
            scope='Lexical candidate inventory only. Exact existing accepted TRAIN source recovery;no target generation,masking,source rewrite,model scoring,DEV/TEST/reserve inspection or quality claim. Private source text is retained only for subsequent explicit eligibility inspection.')
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(plan_sha256=file_sha256(out/'plan.json'),wall_seconds=time.time()-started,
            assigned_chips=0,research_optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        print(json.dumps({k:result[k] for k in ['status','wall_seconds']}),flush=True)


if __name__=='__main__':main()
