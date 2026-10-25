"""Larger accepted-TRAIN pool with full original-cohort equivalence gate."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sqlite3
import time

from compute_trainium.data import content_hash,source_group,split_for_group
from compute_trainium.source_arithmetic import normalize
from compute_trainium.source_arithmetic_fast import original_chains
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='natural-source-chain-pool-v1'
PUBLIC=Path('results/natural_source_chain_pool_20261025')


def canonical(expression,blind=False):
    text=normalize(expression);tree=ast.parse(text,mode='eval')
    def visit(node):
        if isinstance(node,ast.Constant):return ('number','_' if blind else str(Fraction(ast.get_source_segment(text,node))))
        if isinstance(node,ast.UnaryOp):return (type(node.op).__name__,visit(node.operand))
        if isinstance(node,ast.BinOp):return (type(node.op).__name__,visit(node.left),visit(node.right))
        raise ValueError('Only checked arithmetic trees may be grouped')
    return hashlib.sha256(json.dumps(visit(tree.body),separators=(',',':')).encode()).hexdigest()


def guard(root,deadline):
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if time.time()>deadline or available<512*1024**3 or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:
        raise RuntimeError('Finite source-pool preparation resource cap')


def scan(task):
    import pyarrow.parquet as pq
    from scripts import audit_natural_source_chains as parent
    parent.original_chains=original_chains
    root,item,selected,deadline=task;root=Path(root)
    guard(root,deadline);file=root/item['relative']
    if file_sha256(file)!=item['sha256']:raise ValueError('Pinned raw source changed')
    by_url=defaultdict(dict)
    for row in selected:by_url[row['url']][row['id']]=row
    seen=set();records=[]
    for batch in pq.ParquetFile(file).iter_batches(batch_size=128,columns=['url','text'],use_threads=False):
        guard(root,deadline)
        for row in batch.to_pylist():
            allowed=by_url.get(row['url'])
            if not allowed:continue
            key=content_hash(row['text'])
            if key not in allowed or key in seen:continue
            meta=allowed[key];group=source_group('math',row['url'])
            if split_for_group(group)!='train' or hashlib.sha256(group.encode()).hexdigest()!=meta['group']:
                raise AssertionError('Accepted TRAIN source identity changed')
            seen.add(key);record=parent.candidate({**meta,'text':row['text']})
            if record:records.append(record)
    return list(seen),records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();started=time.time();out=args.root/IDENTITY;out.mkdir()
    previous=args.root/'natural-source-chain-audit-v1';prior=json.loads((previous/'report.json').read_text())
    if prior['status']!='COMPLETE_NATURAL_SOURCE_CHAIN_FEASIBILITY_AUDIT':raise ValueError('Complete original audit required')
    cohort=args.root/'natural-worked-reasoning-audit-v1';cohort_plan=json.loads((cohort/'plan.json').read_text())
    selected=json.loads((cohort/'selected-train-metadata.json').read_text());original_ids={r['id'] for r in selected}
    expected={r['id']:r for r in json.loads((previous/'source-span-candidates.json').read_text())}
    database=args.root/'natural-append-6500m-v2/extension/documents.sqlite'
    connection=sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True)
    rows=connection.execute("select id,source_group,source,shard,length from documents where domain='math' and split='train' and shard>=18 order by id").fetchall();connection.close()
    population=[dict(id=i,group=g,url=u,shard=s,source_tokens=n) for i,g,u,s,n in rows]
    sources=['scripts/prepare_natural_source_chain_pool.py','scripts/audit_natural_source_chains.py',
        'compute_trainium/source_arithmetic.py','compute_trainium/source_arithmetic_fast.py',
        'tests/test_source_arithmetic.py','tests/test_source_chain_equivalence.py','docs/NATURAL_SOURCE_CHAIN_POOL_20261025.md']
    plan=dict(identity=IDENTITY,registered_unix=started,seconds_limit=1200,maximum_readers=4,
        source_sha256={p:file_sha256(p) for p in sources},accepted_train_documents=len(population),
        accepted_metadata_sha256=hashlib.sha256(json.dumps(population,separators=(',',':')).encode()).hexdigest(),
        database_sha256=file_sha256(database),original_report_sha256=file_sha256(previous/'report.json'),
        original_candidates_sha256=prior['candidates_sha256'],original_cohort_documents=len(original_ids),
        inputs=cohort_plan['inputs'],scope=Path(sources[-1]).read_text(),assigned_chips=0,research_optimizer_updates=0,test_accessed=False)
    PUBLIC.mkdir(parents=True,exist_ok=False);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    write_state(out/'accepted-train-metadata.json',population)
    result=dict(status='STOPPED_NATURAL_SOURCE_CHAIN_POOL');seen=set();records=[]
    try:
        deadline=started+1200;guard(out,deadline)
        tasks=[(str(args.root),item,[r for r in population if r['shard']==item['shard']],deadline) for item in plan['inputs']]
        with ProcessPoolExecutor(4,mp_context=mp.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(scan,task) for task in tasks]):
                identities,examples=future.result();ids=set(identities)
                if seen&ids:raise AssertionError('Duplicate accepted identity')
                seen.update(ids);records.extend(examples)
                write_state(out/'progress.json',dict(recovered=len(seen),chain_candidates=len(records),wall_seconds=time.time()-started))
        if seen!={r['id'] for r in population}:raise AssertionError('Accepted source reconstruction incomplete')
        actual={r['id']:r for r in records if r['id'] in original_ids}
        if actual!=expected:raise AssertionError('Accelerated extraction changed original cohort candidates/spans/fragments')
        for row in records:
            expression=row['original_source_fragments'][0]
            row['arithmetic_ast_group']=canonical(expression);row['operation_tree_group']=canonical(expression,True)
        records.sort(key=lambda r:r['id']);write_state(out/'source-span-candidates.json',records)
        if any(file_sha256(p)!=h for p,h in plan['source_sha256'].items()):raise AssertionError('Frozen checker/source changed')
        result=dict(status='COMPLETE_NATURAL_SOURCE_CHAIN_POOL_FEASIBILITY',accepted_train_documents=len(population),recovered_documents=len(seen),
            source_chain_documents=len(records),source_groups=len({r['group'] for r in records}),
            arithmetic_ast_groups=len({r['arithmetic_ast_group'] for r in records}),operation_tree_groups=len({r['operation_tree_group'] for r in records}),
            premise_operation_counts=dict(Counter(r['source_spans']['original_operations'] for r in records)),
            original_cohort_documents_checked=len(original_ids),original_candidate_records_equal=True,
            original_positive_candidates=len(expected),candidates_sha256=file_sha256(out/'source-span-candidates.json'),
            training_views_created=0,model_predictions_read=0,
            scope='Source-span candidate reconstruction only. Fast/original extraction identical on all original20000 documents,including negatives. All inputs and intermediate/answer fragments come from accepted TRAIN. Canonical AST groups are metadata,not rewritten data. No teacher/student model,held-out access,training admission or general-reasoning claim.')
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(plan_sha256=file_sha256(out/'plan.json'),wall_seconds=time.time()-started,assigned_chips=0,
            research_optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        print(json.dumps({k:result[k] for k in ['status','wall_seconds']}),flush=True)


if __name__=='__main__':main()
