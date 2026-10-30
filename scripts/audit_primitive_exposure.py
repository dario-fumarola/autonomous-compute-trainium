"""Source-only elementary exposure census on previously frozen TRAIN shards."""
import argparse
from collections import Counter
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

ID = 'primitive-exposure-census-v1'
PARENT = 'latent-exposure-source-audit-v1'
PUBLIC = Path('results/primitive_arithmetic_20261030/exposure-v1')


def worker(root, rank):
    import numpy as np
    from tokenizers import Tokenizer
    from compute_trainium.authored_primitive_arithmetic import equalities
    from compute_trainium.direct_arithmetic_panel import expression_key
    from compute_trainium.natural_code_context import token_span
    from compute_trainium.source_arithmetic import Unsupported
    out = root/ID; p = read(out/'plan.json'); started = time.time()
    guard(root,p)
    if any(file_sha256(f) != h for f,h in p['source_sha256'].items()):
        raise AssertionError('Frozen census source changed')
    tokenizer = Tokenizer.from_file(p['tokenizer_path'])
    tokens = np.memmap(root/p['source_relative']/'math.u16', dtype='<u2', mode='r')
    counts = Counter(); exposure = {k:Counter() for k in p['endpoints']}
    keys = {k:set() for k in p['endpoints']}
    for shard in (rank,rank+4):
        path = root/PARENT/f'PRIVATE-documents-{shard}.json'
        if file_sha256(path) != p['shard_sha256'][str(shard)]:
            raise AssertionError('Original TRAIN document shard changed')
        for i,row in enumerate(read(path)):
            if i % 100 == 0: guard(root,p)
            counts['math_documents'] += 1
            length = min(row['source_tokens'],p['maximum_virtual_position']-row['start'])
            ids = tokens[row['source_offset']:row['source_offset']+length].tolist()
            text = tokenizer.decode(ids,skip_special_tokens=False)
            if '=' not in text: continue
            try: matches = list(equalities(text))
            except Unsupported:
                counts['unsupported_document_length'] += 1; continue
            if not matches: continue
            counts['verified_equalities'] += len(matches)
            encoding = tokenizer.encode(text,add_special_tokens=False)
            if encoding.ids != ids:
                counts['matched_document_roundtrip_rejections'] += 1; continue
            for match in matches:
                target = token_span(text,encoding.offsets,match['target_start'],match['target_end'])
                premise = token_span(text,encoding.offsets,match['premise_start'],match['premise_end'])
                if target is None or premise is None:
                    counts['alignment_rejections'] += 1; continue
                key = expression_key(text[match['premise_start']:match['premise_end']])
                for label,budget in p['endpoints'].items():
                    if row['start']+target[1] > budget['virtual_position']: continue
                    c = exposure[label]; c['aligned'] += 1; c['answer_tokens'] += target[1]-target[0]
                    context_start = ((row['start']+target[0])//512)*512
                    if row['start']+1+premise[0] >= context_start:
                        c['intact'] += 1; c['intact_'+match['stratum']] += 1
                        c['intact_answer_tokens'] += target[1]-target[0]; keys[label].add(key)
            if i % 2000 == 0:
                write_state(out/f'progress-{rank}.json',dict(documents=counts['math_documents'],
                            wall_seconds=time.time()-started))
    write_state(out/f'PRIVATE-expression-keys-{rank}.json',{k:sorted(v) for k,v in keys.items()})
    u = resource.getrusage(resource.RUSAGE_SELF)
    write_state(out/f'worker-{rank}.json',dict(status='COMPLETE_PRIMITIVE_EXPOSURE_WORKER',
        counts=dict(counts),exposure={k:dict(v) for k,v in exposure.items()},
        wall_seconds=time.time()-started,cpu_seconds=u.ru_utime+u.ru_stime,peak_rss_bytes=u.ru_maxrss*1024))


def run(root):
    out = root/ID; started = time.time()
    if out.exists() or PUBLIC.exists(): raise FileExistsError('Preserve every attempt')
    parent = read(root/PARENT/'plan.json'); receipt = read(root/PARENT/'report.json')
    panel = root/'primitive-arithmetic-panel-v1/PRIVATE-panel.json'
    panel_report = read(panel.parent/'report.json')
    if receipt['status'] != 'COMPLETE_EXPOSURE_SAFE_SOURCE_AUDIT':
        raise AssertionError('Original source audit required')
    if file_sha256(panel) != panel_report['files'][panel.name]:
        raise AssertionError('Frozen primitive panel changed')
    files = ['scripts/audit_primitive_exposure.py','scripts/audit_latent_exposure_sources.py',
        'compute_trainium/authored_primitive_arithmetic.py','compute_trainium/authored_direct_arithmetic.py',
        'compute_trainium/direct_arithmetic_panel.py','compute_trainium/source_arithmetic.py',
        'compute_trainium/natural_code_context.py','docs/PRIMITIVE_EXPOSURE_CENSUS_20261030.md']
    p = {k:parent[k] for k in ('endpoints','maximum_virtual_position','source_relative','math_sha256',
                              'tokenizer_path','tokenizer_sha256','index_report_sha256','recipe_sha256')}
    p.update(identity=ID,registered_unix=started,absolute_deadline_unix=started+1200,workers=4,
        source_sha256={f:file_sha256(f) for f in files},parent_report_sha256=file_sha256(root/PARENT/'report.json'),
        shard_sha256={str(i):receipt['files'][f'PRIVATE-documents-{i}.json'] for i in range(8)},
        primitive_panel_sha256=file_sha256(panel),assigned_chips=0,optimizer_updates=0,model_scores=0,
        terminal_alpha_spent=0,automatic_promotion=False,
        scope=Path('docs/PRIMITIVE_EXPOSURE_CENSUS_20261030.md').read_text())
    guard(root,p);out.mkdir(mode=0o700);PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',{k:v for k,v in p.items() if k!='tokenizer_path'})
    result = dict(status='STOPPED_PRIMITIVE_EXPOSURE_CENSUS');jobs=[];logs=[]
    try:
        if (file_sha256(root/p['source_relative']/'math.u16') != p['math_sha256']
                or file_sha256(p['tokenizer_path']) != p['tokenizer_sha256']):
            raise AssertionError('Immutable source/tokenizer changed')
        for rank in range(4):
            log = (out/f'worker-{rank}.log').open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.audit_primitive_exposure',
                '--root',str(root),'--worker',str(rank)],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'TOKENIZERS_PARALLELISM':'false','USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            guard(root,p)
            if any(j.poll() not in (None,0) for j in jobs): raise RuntimeError('Source census worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs): raise RuntimeError('Source census worker failed')
        receipts = [read(out/f'worker-{i}.json') for i in range(4)]
        if any(r['status'] != 'COMPLETE_PRIMITIVE_EXPOSURE_WORKER' for r in receipts):
            raise AssertionError('Four completed workers required')
        keys = [read(out/f'PRIVATE-expression-keys-{i}.json') for i in range(4)]
        cases = read(panel); summary = {}
        for label in p['endpoints']:
            known = set().union(*(set(v[label]) for v in keys))
            row = dict(sum((Counter(r['exposure'][label]) for r in receipts),Counter()))
            row.update(unique_intact_expressions=len(known),
                primitive_panel_exact_expression_matches=sum(v['expression_key'] in known for v in cases),
                primitive_panel_matches_by_operator={k:sum(v['stratum']==k and v['expression_key'] in known for v in cases)
                    for k in ('add','subtract','multiply','divide')})
            summary[label] = row
        result.update(status='COMPLETE_PRIMITIVE_EXPOSURE_CENSUS',
            counts=dict(sum((Counter(r['counts']) for r in receipts),Counter())),exposure=summary,workers=receipts,
            private_key_sha256={f.name:file_sha256(f) for f in out.glob('PRIVATE-*.json')})
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        for j in jobs:
            if j.poll() is None: j.terminate()
        for j in jobs:
            try: j.wait(timeout=10)
            except subprocess.TimeoutExpired: j.kill();j.wait()
        for log in logs: log.close()
        result.update(wall_seconds=time.time()-started,assigned_chips=0,optimizer_updates=0,model_scores=0,
            protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False,
            plan_sha256=file_sha256(out/'plan.json'),scope=p['scope'])
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('workers','scope','private_key_sha256')}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int)
    a=p.parse_args();run(a.root) if a.worker is None else worker(a.root,a.worker)
