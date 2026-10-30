"""Prospective source-only panel and exact training-prefix exposure census."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import shutil
import sqlite3
import time

import numpy as np
from tokenizers import Tokenizer

from compute_trainium.authored_direct_arithmetic import direct_equalities
from compute_trainium.direct_arithmetic_panel import eligible_cases,expression_key,choose_panel
from compute_trainium.natural_code_context import token_span
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID='direct-arithmetic-panel-v1'
PUBLIC=Path('results/direct_arithmetic_20261030/source-preparation-v1')


def read(p):return json.loads(p.read_text())


def run(root):
    start=time.time();out=root/ID
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every source preparation')
    index=root/'indexed-natural-sources-v1';accepted=read(index/'report.json');old=read(index/'plan.json')
    if accepted['status']!='PASS_INDEXED_NATURAL_SOURCES' or accepted['protected_group_overlap']:
        raise AssertionError('Accepted indexed sources required')
    source=root/accepted['source_relative'];db=source/'documents.sqlite'
    fitpath=root/'interleaved-mixed-round-tuning-v1/plan.json';fit=read(fitpath)
    endpoint=fit['arms']['mixed-lr1']['checkpoint_updates'][-1]*fit['world']*fit['batch']*512
    if fit['data_relative']!=index.name or fit['model_config']['max_seq']!=512:
        raise AssertionError('Exact scientific source prefix required')
    tokenizer_path=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    files=['scripts/prepare_direct_arithmetic_panel.py','compute_trainium/direct_arithmetic_panel.py',
        'compute_trainium/authored_direct_arithmetic.py','compute_trainium/source_arithmetic.py',
        'compute_trainium/natural_code_context.py','tests/test_direct_arithmetic_panel.py',
        'tests/test_authored_direct_arithmetic.py','docs/DIRECT_ARITHMETIC_DIAGNOSTIC_20261030.md']
    plan=dict(identity=ID,registered_unix=start,absolute_deadline_unix=start+1200,
        maximum_heldout_documents=100000,maximum_rss_bytes=8*1024**3,
        source_sha256={f:file_sha256(f) for f in files},index_report_sha256=file_sha256(index/'report.json'),
        database_sha256=old['database_sha256'],training_recipe_sha256=accepted['files']['PRIVATE-training-documents.jsonl'],
        math_tokens_sha256=accepted['raw_token_sha256']['math'],tokenizer_sha256=file_sha256(tokenizer_path),
        fit_plan_sha256=file_sha256(fitpath),scientific_training_last_virtual_position=endpoint,
        excluded_group_files=old['excluded_group_files'],quota_per_bin=128,minimum_per_bin=32,
        source_context=256,maximum_answer_tokens=16,assigned_chips=0,optimizer_updates=0,
        model_scores=0,confirmation_reserves_accessed=False,terminal_alpha_spent=0,
        scope='New development diagnostic only. Source-group bucket excluded from all indexed fits; '
              'fixed authentic equations, no model-dependent selection. Exact verified-equation dedup against '
              'actual scientific math training prefix, not a universal pretraining-contamination guarantee. No fit admission.')
    out.mkdir(mode=0o700);PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    counts=Counter();train=Counter();seen_expressions=set();seen_shapes=set();candidates=[]
    result=dict(status='STOPPED_DIRECT_ARITHMETIC_PREPARATION');connection=None

    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (time.time()>plan['absolute_deadline_unix'] or available<512*1024**3
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>plan['maximum_rss_bytes']
                or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3):
            raise RuntimeError('Finite source-preparation reserve / duration bound')

    try:
        guard()
        recipe=index/'PRIVATE-training-documents.jsonl'
        if (file_sha256(db)!=plan['database_sha256'] or file_sha256(recipe)!=plan['training_recipe_sha256']
                or file_sha256(source/'math.u16')!=plan['math_tokens_sha256']):
            raise AssertionError('Original source bytes changed')
        excluded=set()
        for relative,digest in plan['excluded_group_files'].items():
            path=root/relative
            if file_sha256(path)!=digest:raise AssertionError('Protected source registry changed')
            excluded.update(r['group'] for r in read(path))
        tokenizer=Tokenizer.from_file(str(tokenizer_path));tokens=np.memmap(source/'math.u16',dtype='<u2',mode='r')
        training_groups=set()
        with recipe.open() as stream:
            for line in stream:
                meta=json.loads(line)
                if meta['start']>=endpoint:break
                training_groups.add(meta['group'])
                if meta['domain']!='math':continue
                guard();train['math_documents']+=1
                n=min(meta['source_tokens'],endpoint-meta['start']);lo=meta['source_offset']
                ids=tokens[lo:lo+n].tolist();text=tokenizer.decode(ids,skip_special_tokens=False)
                encoding=tokenizer.encode(text,add_special_tokens=False)
                if encoding.ids!=ids:
                    train['roundtrip_excluded']+=1;continue
                matches=list(direct_equalities(text));train['verified_calculations']+=len(matches)
                for match in matches:
                    expr=text[match['premise_start']:match['premise_end']]
                    seen_expressions.add(expression_key(expr));seen_shapes.add(expression_key(expr,blind=True))
                    target=token_span(text,encoding.offsets,match['target_start'],match['target_end'])
                    premise=token_span(text,encoding.offsets,match['premise_start'],match['premise_end'])
                    if target is None or premise is None:continue
                    train['aligned_answer_tokens']+=target[1]-target[0]
                    virtual_target=meta['start']+1+target[0]
                    window_start=((virtual_target-1)//512)*512
                    if meta['start']+1+premise[0]>=window_start:
                        train['packed_premise_calculations']+=1
                        train['packed_premise_answer_tokens']+=target[1]-target[0]
                if train['math_documents']%2000==0:
                    write_state(out/'progress.json',dict(stage='training_census',documents=train['math_documents'],seconds=time.time()-start))
        if any(int(g[:8],16)%10==0 for g in training_groups):raise AssertionError('Training source-group exclusion failed')
        write_state(out/'PRIVATE-training-expression-keys.json',dict(expressions=sorted(seen_expressions),shapes=sorted(seen_shapes)))
        connection=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
        cursor=connection.execute("select id,source_group,offset,length from documents where domain='math' and split='train' order by id")
        for document_id,group,lo,length in cursor:
            if int(group[:8],16)%10!=0 or group in excluded:continue
            if counts['documents']>=plan['maximum_heldout_documents']:break
            guard()
            if group in training_groups:raise AssertionError('Source-group overlap')
            counts['documents']+=1;n=min(length,16384)
            ids=tokens[lo:lo+n].tolist();text=tokenizer.decode(ids,skip_special_tokens=False)
            encoding=tokenizer.encode(text,add_special_tokens=False)
            if encoding.ids!=ids:
                counts['roundtrip_excluded']+=1;continue
            counts['roundtrip_exact']+=1
            for row in eligible_cases(text,encoding):
                counts['source_eligible']+=1
                if row['expression_key'] in seen_expressions:
                    counts['training_expression_exclusion']+=1;continue
                candidates.append(dict(document_id=document_id,group=group,source_offset=lo,source_tokens=n,
                    seen_training_shape=row['shape_key'] in seen_shapes,**row))
            if counts['documents']%2000==0:
                write_state(out/'progress.json',dict(stage='development_source_scan',documents=counts['documents'],
                    candidates=len(candidates),seconds=time.time()-start))
        selected,bins,ready=choose_panel(candidates,quota=plan['quota_per_bin'],minimum=plan['minimum_per_bin'])
        if not ready:
            result.update(status='INSUFFICIENT_DIRECT_ARITHMETIC_SOURCE_COVERAGE',counts=dict(counts),bins=bins)
            return
        rows=[]
        for i,row in enumerate(selected):
            lo=row['source_offset'];a,b=row['target_tokens'];begin=row['window_start']
            seq=tokens[lo+begin:lo+b].astype(np.int64).tolist()
            shift=1 if begin==0 else 0
            if shift:seq=[1]+seq
            first=a-begin+shift
            if first<1 or len(seq)-1>257 or len(seq)-first>16:raise AssertionError('Exact original prefix / target bounds')
            rows.append(dict(index=i,**row,tokens=seq,answer_start=first))
        write_state(out/'PRIVATE-panel.json',rows)
        write_state(out/'PRIVATE-candidates.json',candidates)
        if any(file_sha256(f)!=h for f,h in plan['source_sha256'].items()):raise AssertionError('Frozen preparation code changed')
        guard()
        result.update(status='FROZEN_DIRECT_ARITHMETIC_DEVELOPMENT_PANEL',counts=dict(counts),bins=bins,
            cases=len(rows),source_groups=len({r['group'] for r in rows}),unique_expressions=len({r['expression_key'] for r in rows}),
            seen_training_shapes=sum(r['seen_training_shape'] for r in rows),
            answer_tokens=sum(len(r['tokens'])-r['answer_start'] for r in rows),
            maximum_input_tokens=max(len(r['tokens'])-1 for r in rows),source_group_overlap=0,
            verified_training_expression_overlap=0,protected_group_overlap=0,
            files={f.name:file_sha256(f) for f in out.glob('PRIVATE-*.json')})
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        if connection is not None:connection.close()
        result.update(training_census=dict(train),training_unique_expressions=len(seen_expressions),
            training_unique_shapes=len(seen_shapes),training_real_targets=fit['arms']['mixed-lr1']['real_targets'],
            wall_seconds=time.time()-start,assigned_chips=0,optimizer_updates=0,model_scores=0,
            protected_groups_scored=0,terminal_alpha_spent=0,plan_sha256=file_sha256(out/'plan.json'),scope=plan['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
