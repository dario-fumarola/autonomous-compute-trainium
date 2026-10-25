"""New development-only original arithmetic;protected group metadata only."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import shutil
import sqlite3
import time
from tokenizers import Tokenizer
from compute_trainium.data import content_hash,source_group,split_for_group
from compute_trainium.natural_chain_views import conclusion_view,normalized_answer
from compute_trainium.source_arithmetic_fast import original_chains
from compute_trainium.source_problem_groups import literal_operator_group
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.prepare_natural_source_chain_pool import canonical

IDENTITY='natural-chain-discovery-dev-v2'
PUBLIC=Path('results/natural_chain_discovery_dev_20261025/extension-v2')


def guard(root,deadline):
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if time.time()>deadline or available<512*1024**3 or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Discovery DEV preparation reserve/cap')


def scan(task):
    import pyarrow.parquet as pq
    from scripts import audit_natural_source_chains as checker
    checker.original_chains=original_chains
    root,item,excluded,known_ast,known_cases,deadline=task;root=Path(root);stats=Counter();found=[]
    file=root/item['relative'];guard(root,deadline)
    if file_sha256(file)!=item['sha256']:raise AssertionError('Pinned raw shard changed')
    tokenizer=Tokenizer.from_file('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    for batch in pq.ParquetFile(file).iter_batches(batch_size=128,columns=['url','text'],use_threads=False):
        guard(root,deadline);urls=batch.column(0);texts=batch.column(1)
        for i in range(len(batch)):
            url=urls[i].as_py()
            try:group=source_group('math',url)
            except ValueError:stats['invalid_url']+=1;continue
            if split_for_group(group)!='dev':stats['non_dev_host_skipped_before_content']+=1;continue
            group_hash=hashlib.sha256(group.encode()).hexdigest()
            if group_hash in excluded:stats['protected_group_skipped_before_content']+=1;continue
            text=texts[i].as_py();stats['new_dev_documents_checked']+=1
            record=checker.candidate(dict(id=content_hash(text),group=group_hash,shard=item['shard'],text=text))
            if not record:continue
            stats['source_chain_candidates']+=1;expression=record['original_source_fragments'][0]
            ast_key=canonical(expression);case_key=literal_operator_group(expression)
            if ast_key in known_ast or case_key in known_cases:stats['prior_train_case_overlap']+=1;continue
            try:
                normalized_answer(record['original_source_fragments'][-1]);views=[conclusion_view(record,tokenizer,k) for k in range(len(record['original_source_fragments'])-1)]
            except ValueError as error:stats[str(error)]+=1;continue
            record.update(arithmetic_ast_group=ast_key,operation_tree_group=canonical(expression,True),conservative_case_group=case_key)
            record['views']=[dict(visible_steps=v.visible_steps,prompt=v.prompt,target=v.target,inputs=v.inputs,targets=v.targets,
                answer_mask=v.target_mask,termination_mask=v.termination_mask,answer_ids=v.answer_ids,prefix_ids=v.prefix_ids) for v in views]
            found.append(record)
    return found,stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time();out=a.root/IDENTITY;out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False)
    pool=a.root/'natural-source-chain-pool-v1';parent=json.loads((pool/'report.json').read_text());train_file=pool/'source-span-candidates.json'
    if file_sha256(train_file)!=parent['candidates_sha256']:raise AssertionError('Original TRAIN candidate file changed')
    train=json.loads(train_file.read_text());known_ast={r['arithmetic_ast_group'] for r in train};known_cases={literal_operator_group(r['original_source_fragments'][0]) for r in train};excluded={r['group'] for r in train}
    databases=[Path('/run/user/1000/autonomous-compute/natural-v2-1050m/documents.sqlite')]+sorted(a.root.glob('natural-append-*/extension/documents.sqlite'))
    database_hashes=[]
    for path in databases:
        db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);excluded.update(r[0] for r in db.execute('select distinct source_group from documents'));db.close();database_hashes.append(file_sha256(path))
    metadata_files=[p for directory in [a.root/'variable-depth-fresh-v1',a.root/'fresh-confirmation-reserve-v3'] for p in directory.glob('*.json') if p.name in ['dev.json','test-sealed.json','private-candidate-pool.json']]
    metadata_hashes={}
    for path in metadata_files:
        # Only JSON group-hash fields are extracted;no tokens/targets decoded.
        excluded.update(x.decode() for x in re.findall(rb'"group"\s*:\s*"([a-f0-9]{64})"',path.read_bytes()));metadata_hashes[str(path.relative_to(a.root))]=file_sha256(path)
    old=a.root/'natural-chain-discovery-dev-v1';old_result=json.loads((old/'report.json').read_text())
    if old_result['status']!='INADEQUATE_NATURAL_CHAIN_DISCOVERY_DEV' or old_result['student_scores_read']:raise ValueError('Only unscored inadequate pool eligible for data extension')
    manifest=json.loads((a.root/'natural-append-6500m-v2/manifest.json').read_text());inputs=[]
    for shard in range(18):
        locations=[a.root/'natural-append-3250m/raw'/f'math-{shard}.parquet',Path('/run/user/1000/autonomous-compute/natural-v2-1050m')/f'math-{shard}.parquet',Path('/run/user/1000/autonomous-compute/data-v1/math.parquet')]
        expected=manifest['sources']['math'][shard]['sha256'];matched=None
        for location in locations:
            if location.is_file() and file_sha256(location)==expected:matched=location;break
        if matched is None:raise ValueError('No checksum-matched existing raw shard: '+str(shard))
        inputs.append(dict(shard=shard,relative=str(matched.relative_to(a.root)) if matched.is_relative_to(a.root) else str(matched),sha256=expected))
    sources=['scripts/prepare_natural_chain_dev_extension.py','compute_trainium/source_problem_groups.py','tests/test_source_problem_groups.py','docs/NATURAL_CHAIN_DISCOVERY_DEV_EXTENSION_20261025.md','compute_trainium/natural_chain_views.py','compute_trainium/source_arithmetic.py','compute_trainium/source_arithmetic_fast.py','scripts/audit_natural_source_chains.py','scripts/prepare_natural_source_chain_pool.py']
    plan=dict(identity=IDENTITY,registered_unix=started,seconds_limit=1200,source_sha256={p:file_sha256(p) for p in sources},
        inputs=inputs,parent_inadequate_report_sha256=file_sha256(old/'report.json'),original_candidates_sha256=file_sha256(old/'eligible-candidates.json'),train_candidates_sha256=file_sha256(train_file),database_sha256=database_hashes,protected_group_metadata_sha256=metadata_hashes,
        excluded_source_groups=len(excluded),known_train_arithmetic_ast_groups=len(known_ast),known_conservative_cases=len(known_cases),
        maximum_tasks=256,minimum_tasks=64,minimum_source_groups=32,maximum_per_group=2,test_targets_accessed=False,
        scope=Path(sources[3]).read_text(),assigned_chips=0,research_optimizer_updates=0,terminal_alpha_spent=0)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan);result=dict(status='STOPPED_NATURAL_CHAIN_DISCOVERY_DEV');stats=Counter();records=json.loads((old/'eligible-candidates.json').read_text())
    try:
        deadline=started+1200;guard(out,deadline)
        with ProcessPoolExecutor(4,mp_context=mp.get_context('spawn')) as workers:
            for f in as_completed([workers.submit(scan,(str(a.root),item,excluded,known_ast,known_cases,deadline)) for item in inputs]):
                examples,counts=f.result();records.extend(examples);stats.update(counts)
        write_state(out/'eligible-candidates.json',records);chosen=[];seen_ids=set();seen_cases=set();groups=Counter()
        for row in sorted(records,key=lambda r:hashlib.sha256(('chain-discovery-dev-v1:'+r['id']).encode()).hexdigest()):
            if row['id'] in seen_ids or row['conservative_case_group'] in seen_cases or groups[row['group']]>=2:continue
            chosen.append(row);seen_ids.add(row['id']);seen_cases.add(row['conservative_case_group']);groups[row['group']]+=1
            if len(chosen)==256:break
        if set(groups)&excluded or any(r['arithmetic_ast_group'] in known_ast or r['conservative_case_group'] in known_cases for r in chosen):raise AssertionError('Train/protected panel overlap')
        write_state(out/'panel.json',chosen)
        if any(file_sha256(p)!=h for p,h in plan['source_sha256'].items()):raise AssertionError('Frozen discovery selection changed')
        enough=len(chosen)>=64 and len(groups)>=32
        result=dict(status='FROZEN_NATURAL_CHAIN_DISCOVERY_DEV' if enough else 'INADEQUATE_NATURAL_CHAIN_DISCOVERY_DEV',
            eligible_candidates=len(records),tasks=len(chosen),source_groups=len(groups),counts=dict(stats),
            operation_counts=dict(Counter(r['source_spans']['original_operations'] for r in chosen)),
            panel_sha256=file_sha256(out/'panel.json'),train_or_protected_group_overlap=0,train_conservative_case_overlap=0,
            student_scores_read=False,scientific_training_admitted=False,new_panel_role='development_only',test_targets_accessed=False)
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(plan_sha256=file_sha256(out/'plan.json'),wall_seconds=time.time()-started,assigned_chips=0,research_optimizer_updates=0,terminal_alpha_spent=0)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);print(json.dumps({k:v for k,v in result.items() if k!='counts'}),flush=True)


if __name__=='__main__':main()
