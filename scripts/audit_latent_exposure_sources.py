"""Source-only feasibility and expression isolation for a longer exposure test."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID='latent-exposure-source-audit-v1'
PUBLIC=Path('results/latent_exposure_20261030/source-audit-v1')


def read(p):return json.loads(p.read_text())


def endpoint(data,real,global_windows=128):
    low,high=1,data.windows//global_windows
    if data.target_count(0,high*global_windows)<real:raise ValueError('Insufficient accepted source')
    while low<high:
        middle=(low+high)//2
        if data.target_count(0,middle*global_windows)>=real:high=middle
        else:low=middle+1
    return low


def guard(root,p):
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if (time.time()>p['absolute_deadline_unix'] or available<512*1024**3
        or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3
        or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3
        or min(os.statvfs(root).f_favail,os.statvfs('/').f_favail)<10100):
        raise RuntimeError('Source-audit duration/resource bound')


def worker(root,rank):
    import numpy as np
    from tokenizers import Tokenizer
    from compute_trainium.authored_direct_arithmetic import direct_equalities
    from compute_trainium.direct_arithmetic_panel import expression_key
    from compute_trainium.natural_code_context import token_span
    from compute_trainium.source_arithmetic import Unsupported
    out=root/ID;p=read(out/'plan.json');started=time.time()
    if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Frozen source audit changed')
    tokenizer=Tokenizer.from_file(p['tokenizer_path'])
    tokens=np.memmap(root/p['source_relative']/'math.u16',dtype='<u2',mode='r')
    records=read(out/f'PRIVATE-documents-{rank}.json');keys=set();counts=Counter();exposure={k:Counter() for k in p['endpoints']}
    for i,row in enumerate(records):
        guard(root,p);counts['math_documents']+=1
        length=min(row['source_tokens'],p['maximum_virtual_position']-row['start'])
        ids=tokens[row['source_offset']:row['source_offset']+length].tolist()
        text=tokenizer.decode(ids,skip_special_tokens=False)
        if '=' not in text:continue
        try:matches=list(direct_equalities(text))
        except Unsupported:
            counts['parser_length_rejections']+=1;continue
        if not matches:continue
        encoding=tokenizer.encode(text,add_special_tokens=False);aligned=encoding.ids==ids
        if not aligned:counts['matched_document_roundtrip_rejections']+=1
        for match in matches:
            keys.add(expression_key(text[match['premise_start']:match['premise_end']]))
            counts['verified_calculations']+=1
            if not aligned:continue
            target=token_span(text,encoding.offsets,match['target_start'],match['target_end'])
            premise=token_span(text,encoding.offsets,match['premise_start'],match['premise_end'])
            if target is None or premise is None:
                counts['alignment_rejections']+=1;continue
            for label,budget in p['endpoints'].items():
                boundary=budget['virtual_position']
                if row['start']+target[1]>boundary:continue
                c=exposure[label];c['aligned_calculations']+=1;c['answer_tokens']+=target[1]-target[0]
                first_target=row['start']+1+target[0]
                context_start=((first_target-1)//512)*512
                if row['start']+1+premise[0]>=context_start:
                    c['intact_calculations']+=1;c['intact_answer_tokens']+=target[1]-target[0]
        if i%2000==0:write_state(out/f'progress-{rank}.json',dict(documents=i,total=len(records),seconds=time.time()-started))
    write_state(out/f'PRIVATE-expression-keys-{rank}.json',sorted(keys))
    write_state(out/f'worker-{rank}.json',dict(status='COMPLETE_EXPOSURE_SOURCE_WORKER',counts=dict(counts),
        exposure={k:dict(v) for k,v in exposure.items()},wall_seconds=time.time()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def run(root):
    from compute_trainium.execution_isolation import managed_running
    from compute_trainium.indexed_natural_sources import IndexedNaturalSources
    from compute_trainium.model import ModelConfig
    from compute_trainium.interleaved_workspace import matrix_work
    if managed_running():raise RuntimeError('No Native owner may overlap this audit')
    out=root/ID
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every attempt')
    start=time.time();index=root/'indexed-natural-sources-v1';data=IndexedNaturalSources(index)
    ix=read(index/'plan.json');ir=read(index/'report.json')
    old=root/'interleaved-mixed-round-tuning-v1/plan.json';lp=read(old)
    ordinary=root/'interleaved-tied-ordinary-tuning-v1/plan.json';op=read(ordinary)
    n=endpoint(data,1_000_000_000);c=ModelConfig(**lp['model_config']);co=ModelConfig(**op['model_config'])
    work=sum(((n+1)//2 if k==2 else n//2)*64*matrix_work(c,512,k,mode='latent',batch=2)['training_matmul_flops'] for k in (2,3))
    ordinary_step=64*matrix_work(co,512,0,mode='ordinary',batch=2)['training_matmul_flops']
    matched=math.ceil(work/ordinary_step)
    budgets={name:dict(updates=u,virtual_position=u*128*512,real_targets=data.target_count(0,u*128))
             for name,u in [('old128M',1957),('one_billion',n),('ordinary_matched_work',matched)]}
    panel=root/'direct-arithmetic-panel-v1';pr=read(panel/'report.json')
    if pr['status']!='FROZEN_DIRECT_ARITHMETIC_DEVELOPMENT_PANEL':raise AssertionError('Frozen historical source panel required')
    files=['scripts/audit_latent_exposure_sources.py','compute_trainium/authored_direct_arithmetic.py',
        'compute_trainium/direct_arithmetic_panel.py','compute_trainium/source_arithmetic.py',
        'compute_trainium/natural_code_context.py','compute_trainium/indexed_natural_sources.py',
        'compute_trainium/interleaved_workspace.py','compute_trainium/model.py',
        'docs/LATENT_EXPOSURE_SOURCE_AUDIT_20261030.md','tests/test_latent_exposure_sources.py']
    p=dict(identity=ID,registered_unix=start,absolute_deadline_unix=start+1200,workers=8,endpoints=budgets,
        maximum_virtual_position=budgets['ordinary_matched_work']['virtual_position'],
        candidate_training_matrix_work=work,ordinary_step_matrix_work=ordinary_step,
        maximum_training_matrix_work=matched*ordinary_step,
        latent_parent_sha256=file_sha256(old),ordinary_parent_sha256=file_sha256(ordinary),
        source_relative=ir['source_relative'],index_report_sha256=file_sha256(index/'report.json'),
        recipe_sha256=ir['files']['PRIVATE-training-documents.jsonl'],math_sha256=ir['raw_token_sha256']['math'],
        tokenizer_path='/run/user/1000/autonomous-compute/data-v1/tokenizer.json',
        historical_panel_sha256=pr['files']['PRIVATE-panel.json'],excluded_group_files=ix['excluded_group_files'],
        source_sha256={f:file_sha256(f) for f in files},assigned_chips=0,optimizer_updates=0,model_scores=0,
        terminal_alpha_spent=0,scientific_fit_admitted=False)
    p['tokenizer_sha256']=file_sha256(p['tokenizer_path'])
    guard(root,p);out.mkdir(mode=0o700);PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',p)
    public={k:v for k,v in p.items() if k!='tokenizer_path'};write_state(PUBLIC/'plan.json',public)
    result=dict(status='STOPPED_LATENT_EXPOSURE_SOURCE_AUDIT');jobs=[];logs=[]
    try:
        recipe=index/'PRIVATE-training-documents.jsonl'
        if file_sha256(recipe)!=p['recipe_sha256'] or file_sha256(root/p['source_relative']/'math.u16')!=p['math_sha256']:
            raise AssertionError('Immutable source bytes changed')
        if file_sha256(panel/'PRIVATE-panel.json')!=p['historical_panel_sha256']:raise AssertionError('Historical cases changed')
        excluded=set()
        for f,h in p['excluded_group_files'].items():
            if file_sha256(root/f)!=h:raise AssertionError('Protected source registry changed')
            excluded.update(x['group'] for x in read(root/f))
        shards=[[] for _ in range(8)];training_groups=set();documents=0
        with recipe.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['start']>=p['maximum_virtual_position']:break
                if row['group'] in excluded or int(row['group'][:8],16)%10==0:raise AssertionError('Training-source isolation failed')
                training_groups.add(row['group']);documents+=1
                if row['domain']=='math':shards[sum(map(len,shards))%8].append(row)
                if documents%20000==0:guard(root,p)
        for rank,rows in enumerate(shards):
            write_state(out/f'PRIVATE-documents-{rank}.json',rows)
            log=(out/f'worker-{rank}.log').open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.audit_latent_exposure_sources','--root',str(root),'--worker',str(rank)],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'TOKENIZERS_PARALLELISM':'false','USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        del shards
        while any(j.poll() is None for j in jobs):
            guard(root,p)
            if any(j.poll() not in (None,0) for j in jobs):raise RuntimeError('Source census worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs):raise RuntimeError('Source census worker failed')
        receipts=[read(out/f'worker-{i}.json') for i in range(8)]
        if any(v['status']!='COMPLETE_EXPOSURE_SOURCE_WORKER' for v in receipts):raise AssertionError('All census workers required')
        keys=set().union(*(set(read(out/f'PRIVATE-expression-keys-{i}.json')) for i in range(8)))
        historical=read(panel/'PRIVATE-panel.json');selected=[v for v in historical if v['expression_key'] not in keys]
        if any(v['group'] in training_groups|excluded for v in selected):raise AssertionError('Prospective panel source overlap')
        bins=Counter(v['stratum'] for v in selected);ready=all(bins[b]>=32 for b in ('2','3','4-6'))
        counts=sum((Counter(v['counts']) for v in receipts),Counter())
        exposure={k:dict(sum((Counter(v['exposure'][k]) for v in receipts),Counter())) for k in budgets}
        write_state(out/'PRIVATE-all-expression-keys.json',sorted(keys));write_state(out/'PRIVATE-panel.json',selected)
        result.update(status='COMPLETE_EXPOSURE_SAFE_SOURCE_AUDIT' if ready else 'INSUFFICIENT_EXPOSURE_SAFE_CASES',
            training_documents=documents,training_source_groups=len(training_groups),counts=dict(counts),exposure=exposure,
            unique_training_expressions=len(keys),historical_cases=len(historical),removed_expressions=len(historical)-len(selected),
            retained_cases=len(selected),bins=dict(bins),source_overlap=0,expression_overlap=0,protected_overlap=0,
            files={f.name:file_sha256(f) for f in out.glob('PRIVATE-*.json')},workers=receipts)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        for j in jobs:
            if j.poll() is None:j.terminate()
        for j in jobs:
            try:j.wait(timeout=10)
            except subprocess.TimeoutExpired:j.kill();j.wait()
        for log in logs:log.close()
        result.update(wall_seconds=time.time()-start,assigned_chips=0,optimizer_updates=0,model_scores=0,
            scientific_fit_admitted=False,protected_groups_scored=0,terminal_alpha_spent=0,
            plan_sha256=file_sha256(out/'plan.json'),public_plan_sha256=file_sha256(PUBLIC/'plan.json'))
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('files','workers')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int)
    a=p.parse_args();run(a.root) if a.worker is None else worker(a.root,a.worker)
