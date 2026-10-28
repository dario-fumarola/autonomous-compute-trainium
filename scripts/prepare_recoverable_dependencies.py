"""Freeze authentic code/math dependency targets before any model scoring."""
import argparse,ast,collections,concurrent.futures,hashlib,json,time,warnings,shutil
from pathlib import Path
from fractions import Fraction
import re
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from compute_trainium.data import content_hash,source_group
from compute_trainium.natural_code_context import unambiguous_local_uses,token_span
from compute_trainium.natural_chain_views import source_fragments
from compute_trainium.source_arithmetic import check
from compute_trainium.recoverable_dependencies import source_target
from compute_trainium.training import file_sha256
from scripts.prepare_natural import rows


def resource_guard(root,deadline):
 available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
 if available<512*1024**3 or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3 or time.time()>deadline:raise RuntimeError('Source preparation reserve/time guard')


def scan_math(task):
 import pyarrow.parquet as pq
 root,raw,chosen,deadline=task;root=Path(root);resource_guard(root,deadline);tok=AutoTokenizer.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True,trust_remote_code=False);by_group=collections.defaultdict(dict)
 for row in chosen:by_group[row['group']][row['id']]=row
 found=[];cases=[];excluded=collections.Counter()
 for batch in pq.ParquetFile(raw['path']).iter_batches(batch_size=128,columns=['url','text'],use_threads=False):
  resource_guard(root,deadline);urls,texts=batch.column(0),batch.column(1)
  for k in range(len(batch)):
   try:group=hashlib.sha256(source_group('math',urls[k].as_py()).encode()).hexdigest()
   except ValueError:continue
   if group not in by_group:continue
   text=texts[k].as_py();key=content_hash(text)
   if key not in by_group[group]:continue
   row=by_group[group].pop(key);found.append(key)
   if hashlib.sha256(text.encode()).hexdigest()!=row['original_utf8_sha256']:raise AssertionError('Original mathematical source bytes changed')
   parts=source_fragments(row);spans=[row['source_spans']['expression'],*row['source_spans']['intermediate_steps'],row['source_spans']['conclusion']]
   if [text[a:b] for a,b in spans]!=parts:raise AssertionError('Original mathematical fragment alignment changed')
   a,b=spans[-1]
   try:
    case=source_target(text,a,b,tok,kind='math_conclusion',identity=key,group=group,prefix_cap=128,short_tokens=1,antecedent_start=spans[0][0]);case.update(original_operations=row['source_spans']['original_operations'],last_step_operations=row['source_spans']['step_operations'][-1]);cases.append(case)
   except ValueError as error:excluded['math_conclusion:'+str(error)]+=1
   # Exploratory: new numeral in the first source-authored simplification.
   prior=check(parts[0]).literals;lo,hi=spans[1]
   for m in re.finditer(r'\d+(?:\.\d+)?',text[lo:hi]):
    if Fraction(m.group()) in {abs(v) for v in prior}:continue
    try:
     c=source_target(text,lo+m.start(),lo+m.end(),tok,kind='math_intermediate_exploratory',identity=key,group=group,prefix_cap=128,short_tokens=1,antecedent_start=spans[0][0]);cases.append(c)
    except ValueError as error:excluded['math_intermediate:'+str(error)]+=1
    break
 return found,cases,dict(excluded)


def run(root):
 started=time.time();out=root/'recoverable-dependency-panel-v1';public=Path('results/recoverable_dependencies_20261028');out.mkdir(mode=0o700);public.mkdir();tok=AutoTokenizer.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True,trust_remote_code=False);oldtok=Tokenizer.from_file('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
 cp=root/'natural-code-context-v1/panel.json';mp=root/'natural-chain-discovery-dev-v2/panel.json';restore=root/'dependency-source-restore-v1/report.json';rr=json.loads(restore.read_text())
 if rr['status']!='PASS_PINNED_DEPENDENCY_SOURCE_RESTORATION':raise AssertionError('Every original math source must be recovered first')
 code=json.loads(cp.read_text())['records'];math=json.loads(mp.read_text());metadata={x['id']:x for x in json.loads((root/'variable-depth-fresh-v1/dev.json').read_text())};wanted=collections.defaultdict(list)
 for row in code:wanted[row['document_id']].append(row)
 rawmap=json.loads((root/'natural-append-capacity-v4/raw-source-map.json').read_text());code_raw=next(v for v in rawmap if v['domain']=='code');plan=dict(identity=out.name,registered_unix=started,seconds_limit=1200,code_parent_sha256=file_sha256(cp),math_parent_sha256=file_sha256(mp),restoration_report_sha256=file_sha256(restore),code_original_source_sha256=code_raw['receipt']['sha256'],source_sha256={n:file_sha256(n) for n in ['scripts/prepare_recoverable_dependencies.py','compute_trainium/recoverable_dependencies.py','compute_trainium/natural_code_context.py','compute_trainium/natural_chain_views.py','compute_trainium/source_arithmetic.py']},scope=Path('docs/RECOVERABLE_DEPENDENCY_DIAGNOSTIC_20261028.md').read_text(),optimizer_updates=0,assigned_chips=0,model_scores_seen=False)
 for d in [out,public]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 result=dict(status='STOPPED_DEPENDENCY_SOURCE_PANEL',optimizer_updates=0,assigned_chips=0);cases=[];found=set();counts=collections.Counter()
 try:
  if file_sha256(code_raw['path'])!=code_raw['receipt']['sha256']:raise AssertionError('Original code source changed')
  repos={metadata[k]['source'].strip().lower().removesuffix('.git') for k in wanted}
  for source_index,row in enumerate(rows(Path(code_raw['path']))):
   if source_index%256==0:resource_guard(root,started+1200)
   repo=row.get('repo_name','').strip().lower().removesuffix('.git')
   if repo not in repos:continue
   text=row.get('text',row.get('content',''));key=content_hash(text)
   if key not in wanted or key in found:continue
   old=metadata[key]
   if hashlib.sha256(source_group('code',repo).encode()).hexdigest()!=old['group']:continue
   encoded=oldtok.encode(text,add_special_tokens=False)
   if encoded.ids[:len(old['tokens'])]!=old['tokens']:raise AssertionError('Original code source token prefix changed')
   found.add(key)
   try:
    with warnings.catch_warnings():warnings.simplefilter('ignore',SyntaxWarning);uses=unambiguous_local_uses(text)
   except (SyntaxError,ValueError):raise AssertionError('Previously parsed original code source no longer parses')
   for target in wanted[key]:
    matching=[]
    for use in uses:
     span=token_span(text,encoded.offsets,use['target_start'],use['target_end']);definition=token_span(text,encoded.offsets,use['definition_start'],use['definition_end'])
     if span== (target['target_start_token'],target['target_end_token']) and definition and definition[0]==target['definition_start_token']:matching.append(use)
    if len(matching)!=1:raise AssertionError('Original AST-selected lexical dependency changed')
    use=matching[0]
    try:
     c=source_target(text,use['target_start'],use['target_end'],tok,kind='code_binding',identity=target['id'],group=old['group'],prefix_cap=1008,short_tokens=128,antecedent_start=use['definition_start']);cases.append(c)
    except ValueError as error:counts['code:'+str(error)]+=1
  if found!=set(wanted):raise AssertionError('Original code-source reconstruction incomplete')
  import multiprocessing as mm
  tasks=[]
  for raw in rr['records']:
   selected=[x for x in math if x['shard']==raw['index']]
   if selected:tasks.append((str(root),raw,selected,started+1200))
  math_found=[]
  with concurrent.futures.ProcessPoolExecutor(4,mp_context=mm.get_context('spawn')) as pool:
   for f in concurrent.futures.as_completed([pool.submit(scan_math,t) for t in tasks]):
    ids,new,ex=f.result();math_found+=ids;cases+=new;counts.update(ex);(out/'progress.json').write_text(json.dumps(dict(code_documents=len(found),math_documents=len(math_found),cases=len(cases),wall_seconds=time.time()-started)))
  if set(math_found)!={x['id'] for x in math} or len(math_found)!=len(math):raise AssertionError('Original math-source reconstruction incomplete')
  forbidden={json.loads(s)['group'] for split in ['train','tune'] for s in (root/'workspace-source-views-v2'/(split+'-views.jsonl')).read_text().splitlines()}
  if any(c['group'] in forbidden for c in cases):raise AssertionError('Dependency panel overlaps continuation TRAIN/TUNE groups')
  cases.sort(key=lambda x:(x['kind'],x['cohort'],x['id']));(out/'PRIVATE-cases.json').write_text(json.dumps(cases,separators=(',',':'))+'\n');coverage={}
  for kind in sorted({x['kind'] for x in cases}):
   coverage[kind]={str(cohort):dict(cases=len([x for x in cases if x['kind']==kind and x['cohort']==cohort]),source_groups=len({x['group'] for x in cases if x['kind']==kind and x['cohort']==cohort}),real_targets=sum(len(x['answer']) for x in cases if x['kind']==kind and x['cohort']==cohort)) for cohort in [0,1]}
  adequate=all(all(v['cases']>=32 and v['source_groups']>=12 for v in coverage.get(kind,{}).values()) and len(coverage.get(kind,{}))==2 for kind in ['code_binding','math_conclusion'])
  result.update(status='FROZEN_RECOVERABLE_DEPENDENCY_PANEL' if adequate else 'INSUFFICIENT_PRIMARY_DEPENDENCY_COVERAGE',coverage=coverage,exclusions=dict(counts),original_code_documents_recovered=len(found),original_math_documents_recovered=len(math_found),all_source_bytes_verified=True,all_targets_original_source_substrings=True,continuation_train_tune_group_overlap=0,case_file_sha256=file_sha256(out/'PRIVATE-cases.json'),model_scores_seen=False)
 except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  result.update(wall_seconds=time.time()-started,plan_sha256=file_sha256(out/'plan.json'))
  for d in [out,public]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({k:result[k] for k in ['status','wall_seconds']}),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
