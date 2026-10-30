"""Frozen exploratory two-factor readout; no training or success gate."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from compute_trainium.chunked_position_analysis import make_rows, summarize, compare
from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.readout_fixed_depth_causal import guard,indices_for_worker

ID='chunked-path-diagnostic-v1'
SCREEN='chunked-control-completion-v1'
PUBLIC=Path('results/chunked_persistent_workspace_20261029/path-diagnostic-v1')
FAMILIES=('generated','constant')
VARIANTS={'authentic':(False,False),'source_boundary':(True,False),
          'blocked_memory':(False,True),'both':(True,True)}


def read(p):return json.loads(p.read_text())


def prepare(root):
    result=read(root/SCREEN/'report.json')
    if result['status']!='COMPLETE_CHUNKED_CONTROL_COMPLETION':raise AssertionError('Complete exact comparison required')
    main=read(root/SCREEN/'readout-plan.json');files=dict(main['source_sha256'])
    for f in ('compute_trainium/chunked_path_intervention.py','compute_trainium/chunked_position_analysis.py',
              'scripts/diagnose_chunked_paths.py','tests/test_chunked_path_intervention.py',
              'docs/CHUNKED_PATH_DIAGNOSTIC_20261030.md'):
        files[f]=file_sha256(f)
    now=time.time();p=dict(identity=ID,registered_unix=now,absolute_deadline_unix=now+1200,
        seconds_limit=1200,workers=4,threads=8,maximum_worker_rss_bytes=6*1024**3,
        source_sha256=files,completion_sha256={f:file_sha256(root/SCREEN/f) for f in
        ('report.json','readout-plan.json','compare/report.json','readout/report.json')},
        variants={k:dict(source_boundary=a,block_memory=b) for k,(a,b) in VARIANTS.items()},
        bootstrap_seed=40515,bootstrap_replicates=2000,
        scope='Exploratory two-factor intervention on saved selected checkpoints and all reused DEV. Off-distribution path removal, not a new trained predictor, reasoning result, success gate or efficiency claim. No fit, reselection, fresh reserve or promotion.')
    (root/ID).mkdir();PUBLIC.mkdir(parents=True)
    write_state(root/ID/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def verify(root):
    p=read(root/ID/'plan.json');guard(root,p)
    if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Frozen sources changed')
    if any(file_sha256(root/SCREEN/f)!=h for f,h in p['completion_sha256'].items()):raise AssertionError('Completed evidence changed')
    return p,read(root/SCREEN/'readout-plan.json'),read(root/SCREEN/'report.json')


def worker(a):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from compute_trainium.chunked_path_intervention import readout
    from compute_trainium.chunked_persistent_workspace import ChunkedPersistentWorkspace
    from compute_trainium.model import ModelConfig
    p,main,result=verify(a.root);start=time.time();out=a.root/ID/'readout'
    admission=read(a.root/ID/'start.json');torch.set_num_threads(p['threads']);torch.set_num_interop_threads(1)
    evaluation=a.root/main['evaluation_relative']
    if any(file_sha256(evaluation/f)!=h for f,h in main['evaluation_files'].items()):raise AssertionError('Evaluation bytes changed')
    tokens=np.load(evaluation/'broad-tokens.npy',mmap_mode='r');meta=read(evaluation/'PRIVATE-broad-metadata.json')
    rows={};checks=[]
    for family in FAMILIES:
        arm=main['arms'][result['records']['compare']['selection'][family]['arm']]
        checkpoint=a.root/'runs'/arm['phase']/f"PRIVATE-u{arm['checkpoint_updates'][-1]}.pt"
        if file_sha256(checkpoint)!=admission['checkpoint_sha256'][family]:raise AssertionError('Selected checkpoint changed')
        model=ChunkedPersistentWorkspace(ModelConfig(**main['model_config'])).double().eval()
        saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False);model.load_state_dict(saved['model']);del saved
        mode='latent' if family=='generated' else 'pause'
        with torch.inference_mode():
            for domain in range(3):
                guard(a.root,p);i=domain*128+a.worker*2
                x=torch.from_numpy(np.array(tokens[i:i+1,:-1],dtype=np.int64))
                values={k:readout(model,x,128,mode=mode,source_boundary=s,block_memory=b).log_softmax(-1)
                        for k,(s,b) in VARIANTS.items()}
                cached=model.cached(x,128,1,mode).log_softmax(-1)
                ordinary=model(x,steps=0).log_softmax(-1)
                keep=torch.tensor([i for i in range(512) if (i+1)%128!=0])
                errors=dict(authentic_cached=float((values['authentic']-cached).abs().max()),
                    boundary_only=float((values['source_boundary'][:,keep]-values['authentic'][:,keep]).abs().max()),
                    memory_prefix=float((values['blocked_memory'][:,:128]-values['authentic'][:,:128]).abs().max()),
                    both_ordinary=float((values['both']-ordinary).abs().max()))
                if max(errors.values())>1e-8:raise AssertionError('Intervention identity check failed')
                checks.append(dict(family=family,index=i,errors=errors));del values,cached,ordinary
            for label,(s,b) in VARIANTS.items():
                key=family+'-'+label;rows[key]=[]
                for indices in indices_for_worker(a.worker,p['workers']):
                    guard(a.root,p)
                    if time.time()-admission['started_unix']>p['seconds_limit']:raise TimeoutError('Fixed analysis duration')
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['maximum_worker_rss_bytes']:raise RuntimeError('Worker RSS limit')
                    x=torch.from_numpy(np.array(tokens[indices,:-1],dtype=np.int64));y=torch.from_numpy(np.array(tokens[indices,1:],dtype=np.int64))
                    logits=(model.cached(x,128,1,mode) if label=='authentic' else
                            readout(model,x,128,mode=mode,source_boundary=s,block_memory=b))
                    losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
                    rows[key].extend(make_rows(losses.numpy(),meta,indices));del logits,losses
                    write_state(out/f'worker{a.worker}-progress.json',dict(label=key,rows=len(rows[key]),wall_seconds=time.time()-start))
        del model
    write_state(out/f'PRIVATE-worker{a.worker}-scores.json',rows)
    write_state(out/f'worker{a.worker}-report.json',dict(status='COMPLETE_CHUNKED_PATH_WORKER',checks=checks,
        wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def run(root):
    p,main,result=verify(root)
    if managed_running():raise RuntimeError('Native owner must close first')
    start=time.time();out=root/ID/'readout';out.mkdir(exist_ok=False);hashes={}
    analysis=result['records']['compare']
    for family in FAMILIES:
        name=analysis['selection'][family]['arm'];arm=main['arms'][name]
        checkpoint=root/'runs'/arm['phase']/f"PRIVATE-u{arm['checkpoint_updates'][-1]}.pt"
        h=file_sha256(checkpoint);audit=result['records']['audit-'+name]
        if audit['status']!='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS' or {r['checkpoint_sha256'] for r in audit['rows']}!={h}:
            raise AssertionError('Original mature audit required')
        hashes[family]=h
    write_state(root/ID/'start.json',dict(started_unix=start,checkpoint_sha256=hashes))
    jobs,logs=[],[];report=dict(status='STOPPED_CHUNKED_PATH_DIAGNOSTIC')
    try:
        for rank in range(p['workers']):
            log=(out/f'worker{rank}.log').open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.diagnose_chunked_paths','--root',str(root),
                '--action','worker','--worker',str(rank)],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            guard(root,p)
            if time.time()-start>p['seconds_limit']:raise TimeoutError('Fixed diagnostic duration')
            if any(j.poll() not in (None,0) for j in jobs):raise RuntimeError('CPU worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs):raise RuntimeError('CPU worker failed')
        receipts=[read(out/f'worker{k}-report.json') for k in range(p['workers'])]
        if any(r['status']!='COMPLETE_CHUNKED_PATH_WORKER' for r in receipts) or sum(len(r['checks']) for r in receipts)!=24:
            raise AssertionError('Complete 24 identity-check records required')
        parts=[read(out/f'PRIVATE-worker{k}-scores.json') for k in range(p['workers'])]
        labels=[f+'-'+v for f in FAMILIES for v in VARIANTS]
        by={label:sorted([r for part in parts for r in part[label]],key=lambda r:r['index']) for label in labels}
        scores={label:summarize(rows) for label,rows in by.items()}
        for family in FAMILIES:
            actual=scores[family+'-authentic'];expected=analysis['scores'][family]
            values=[(actual['broad_nll'],expected['broad_nll'])]+[(actual['domains'][d],expected['domains'][d]) for d in ('web','math','code')]
            if any(not math.isclose(a,b,abs_tol=1e-8,rel_tol=0) for a,b in values):raise AssertionError('Authentic scores must reconstruct exactly')
        comparisons={f+'-'+v:compare(by[f+'-'+v],by[f+'-authentic'],seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates'])
                     for f in FAMILIES for v in VARIANTS if v!='authentic'}
        interaction={f:scores[f+'-both']['broad_nll']-scores[f+'-source_boundary']['broad_nll']-
            scores[f+'-blocked_memory']['broad_nll']+scores[f+'-authentic']['broad_nll'] for f in FAMILIES}
        report.update(status='COMPLETE_CHUNKED_PATH_DIAGNOSTIC',scores=scores,intervention_improvements=comparisons,
            loss_scale_interaction=interaction,workers=receipts,checkpoint_sha256=hashes,
            private_score_sha256={f.name:file_sha256(f) for f in out.glob('PRIVATE-worker*-scores.json')})
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        for j in jobs:
            if j.poll() is None:j.terminate()
        for j in jobs:
            try:j.wait(timeout=10)
            except subprocess.TimeoutExpired:j.kill();j.wait()
        for log in logs:log.close()
        report.update(wall_seconds=time.time()-start,assigned_chips=0,optimizer_updates=0,
            protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False,scope=p['scope'])
        write_state(root/ID/'report.json',report);write_state(PUBLIC/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run','worker'],required=True);p.add_argument('--worker',type=int)
    a=p.parse_args();worker(a) if a.action=='worker' else (prepare if a.action=='prepare' else run)(a.root)
