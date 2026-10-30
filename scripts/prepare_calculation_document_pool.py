"""Keep authentic TRAIN documents containing verified, authored calculations."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_latent_exposure_sources import guard, read

ID = 'calculation-document-pool-v1'
PARENT = 'latent-exposure-source-audit-v1'
PUBLIC = Path('results/calculation_document_pool_20261030/source-v1')


def worker(root, rank):
    import numpy as np
    from tokenizers import Tokenizer
    from compute_trainium.authored_primitive_arithmetic import equalities
    from compute_trainium.authored_direct_arithmetic import direct_equalities
    from compute_trainium.direct_arithmetic_panel import expression_key
    from compute_trainium.natural_code_context import token_span
    from compute_trainium.source_arithmetic import Unsupported
    out=root/ID;p=read(out/'plan.json');started=time.time();guard(root,p)
    if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):
        raise AssertionError('Frozen pool source changed')
    tok=Tokenizer.from_file(p['tokenizer_path'])
    tokens=np.memmap(root/p['source_relative']/'math.u16',dtype='<u2',mode='r')
    counts=Counter();selected=[]
    for shard in (rank,rank+4):
        path=root/PARENT/f'PRIVATE-documents-{shard}.json'
        if file_sha256(path)!=p['shard_sha256'][str(shard)]:
            raise AssertionError('Original TRAIN shard changed')
        for i,row in enumerate(read(path)):
            if i%100==0:guard(root,p)
            counts['documents']+=1
            length=min(row['source_tokens'],p['maximum_virtual_position']-row['start'])
            ids=tokens[row['source_offset']:row['source_offset']+length].tolist()
            text=tok.decode(ids,skip_special_tokens=False)
            if '=' not in text:continue
            try: matches=list(equalities(text))+list(direct_equalities(text))
            except Unsupported:
                counts['unsupported_length']+=1;continue
            if not matches:continue
            encoded=tok.encode(text,add_special_tokens=False)
            if encoded.ids!=ids:
                counts['roundtrip_rejections']+=1;continue
            spans=[];seen=set()
            for v in matches:
                target=token_span(text,encoded.offsets,v['target_start'],v['target_end'])
                premise=token_span(text,encoded.offsets,v['premise_start'],v['premise_end'])
                if target is None or premise is None:
                    counts['alignment_rejections']+=1;continue
                if target in seen:raise AssertionError('Overlapping verifier families')
                seen.add(target)
                key=expression_key(text[v['premise_start']:v['premise_end']])
                spans.append(dict(premise_tokens=list(premise),target_tokens=list(target),
                                  operations=v['operations'],expression_key=key))
            if spans:
                selected.append(dict(id=row['id'],group=row['group'],domain='math',
                    source_offset=row['source_offset'],source_tokens=length,calculation_spans=spans))
            if i%2000==0:
                write_state(out/f'progress-{rank}.json',dict(documents=counts['documents'],selected=len(selected),
                            wall_seconds=time.time()-started))
    write_state(out/f'PRIVATE-documents-{rank}.json',selected)
    u=resource.getrusage(resource.RUSAGE_SELF)
    write_state(out/f'worker-{rank}.json',dict(status='COMPLETE_CALCULATION_POOL_WORKER',counts=dict(counts),
        selected_documents=len(selected),wall_seconds=time.time()-started,
        cpu_seconds=u.ru_utime+u.ru_stime,peak_rss_bytes=u.ru_maxrss*1024))


def run(root):
    out=root/ID;started=time.time()
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every attempt')
    parent=read(root/PARENT/'plan.json');receipt=read(root/PARENT/'report.json')
    index=read(root/'indexed-natural-sources-v1/plan.json')
    if receipt['status']!='COMPLETE_EXPOSURE_SAFE_SOURCE_AUDIT':
        raise AssertionError('Completed source-isolated parent required')
    files=['scripts/prepare_calculation_document_pool.py','scripts/audit_latent_exposure_sources.py',
        'compute_trainium/authored_primitive_arithmetic.py','compute_trainium/authored_direct_arithmetic.py',
        'compute_trainium/direct_arithmetic_panel.py','compute_trainium/source_arithmetic.py',
        'compute_trainium/natural_code_context.py','docs/CALCULATION_DOCUMENT_POOL_20261030.md']
    p={k:parent[k] for k in ('maximum_virtual_position','source_relative','math_sha256','tokenizer_path','tokenizer_sha256')}
    p.update(identity=ID,registered_unix=started,absolute_deadline_unix=started+1200,workers=4,
        artifact_allowance_bytes=256*1024**2,source_sha256={f:file_sha256(f) for f in files},
        parent_report_sha256=file_sha256(root/PARENT/'report.json'),
        shard_sha256={str(i):receipt['files'][f'PRIVATE-documents-{i}.json'] for i in range(8)},
        excluded_group_files=index['excluded_group_files'],
        assigned_chips=0,optimizer_updates=0,model_scores=0,scientific_fit_admitted=False,
        scope=Path('docs/CALCULATION_DOCUMENT_POOL_20261030.md').read_text())
    guard(root,p);out.mkdir(mode=0o700);PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',{k:v for k,v in p.items() if k!='tokenizer_path'})
    result=dict(status='STOPPED_CALCULATION_DOCUMENT_POOL');jobs=[];logs=[]
    try:
        if (file_sha256(root/p['source_relative']/'math.u16')!=p['math_sha256']
                or file_sha256(p['tokenizer_path'])!=p['tokenizer_sha256']):
            raise AssertionError('Immutable source/tokenizer changed')
        excluded=set()
        for f,h in p['excluded_group_files'].items():
            if file_sha256(root/f)!=h:raise AssertionError('Protected source registry changed')
            excluded.update(v['group'] for v in read(root/f))
        for rank in range(4):
            log=(out/f'worker-{rank}.log').open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.prepare_calculation_document_pool',
                '--root',str(root),'--worker',str(rank)],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'TOKENIZERS_PARALLELISM':'false','USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            guard(root,p)
            if any(j.poll() not in (None,0) for j in jobs):raise RuntimeError('Source pool worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs):raise RuntimeError('Source pool worker failed')
        receipts=[read(out/f'worker-{i}.json') for i in range(4)]
        if any(v['status']!='COMPLETE_CALCULATION_POOL_WORKER' for v in receipts):
            raise AssertionError('All four workers required')
        documents={}
        for i in range(4):
            for row in read(out/f'PRIVATE-documents-{i}.json'):
                if row['group'] in excluded or int(row['group'][:8],16)%10==0:
                    raise AssertionError('Training/source-reserve overlap')
                if row['id'] in documents and row!=documents[row['id']]:
                    raise AssertionError('Conflicting repeated source document')
                documents[row['id']]=row
        rows=sorted(documents.values(),key=lambda v:hashlib.sha256((v['id']+':calculation-pool-v1').encode()).hexdigest())
        groups=Counter();ops=Counter();expressions=set();answers=0
        for row in rows:
            groups[row['group']]+=row['source_tokens']
            for v in row['calculation_spans']:
                ops[str(v['operations'])]+=1;expressions.add(v['expression_key'])
                answers+=v['target_tokens'][1]-v['target_tokens'][0]
        write_state(out/'PRIVATE-pool.json',rows);guard(root,p)
        size=sum(f.stat().st_size for f in out.iterdir() if f.is_file())
        if size>p['artifact_allowance_bytes']:raise RuntimeError('Frozen index artifact allowance exceeded')
        total=sum(groups.values())
        result.update(status='COMPLETE_AUTHENTIC_CALCULATION_DOCUMENT_POOL',workers=receipts,
            scanned_documents=sum(v['counts']['documents'] for v in receipts),documents=len(rows),
            source_groups=len(groups),original_tokens=total,calculation_counts=dict(ops),
            distinct_expressions=len(expressions),verified_answer_tokens=answers,
            largest_group_token_fraction=max(groups.values(),default=0)/max(1,total),
            top10_group_token_fraction=sum(sorted(groups.values(),reverse=True)[:10])/max(1,total),
            protected_group_overlap=0,held_out_hash_bucket_overlap=0,
            original_document_substrings_only=True,source_bytes_copied=0,
            files={f.name:file_sha256(f) for f in out.glob('PRIVATE-*.json')},index_artifact_bytes=size)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        for j in jobs:
            if j.poll() is None:j.terminate()
        for j in jobs:
            try:j.wait(timeout=10)
            except subprocess.TimeoutExpired:j.kill();j.wait()
        for log in logs:log.close()
        result.update(wall_seconds=time.time()-started,assigned_chips=0,optimizer_updates=0,model_scores=0,
            protected_groups_scored=0,terminal_alpha_spent=0,scientific_fit_admitted=False,
            plan_sha256=file_sha256(out/'plan.json'),scope=p['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('workers','files','scope')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int)
    a=p.parse_args();run(a.root) if a.worker is None else worker(a.root,a.worker)
