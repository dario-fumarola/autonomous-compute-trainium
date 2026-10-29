"""Exact chronological FP64 readout of selected, independently audited arms."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.readout_fixed_depth_causal import guard,indices_for_worker
from scripts.analyze_chunked_persistent import verify,summary
from scripts.run_chunked_persistent_screen import FAMILIES


def worker(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from compute_trainium.chunked_persistent_workspace import ChunkedPersistentWorkspace
    from compute_trainium.model import ModelConfig
    p=verify(args.root,args.plan);guard(args.root,p);start=time.time()
    rank,workers=args.worker,p['exact_readout_workers']
    torch.set_num_threads(p['exact_readout_threads']);torch.set_num_interop_threads(1)
    selected=json.loads((args.root/p['identity']/'select/report.json').read_text())['selection']
    evaluation=args.root/p['evaluation_relative']
    if any(file_sha256(evaluation/f)!=h for f,h in p['evaluation_files'].items()):
        raise AssertionError('Frozen evaluation bytes changed')
    tokens=np.load(evaluation/'broad-tokens.npy',mmap_mode='r')
    meta=json.loads((evaluation/'PRIVATE-broad-metadata.json').read_text())
    if len(tokens)!=384 or len(meta)!=384:raise AssertionError('Complete original panel required')
    rows,checks,hashes=[],[],{}
    for family in FAMILIES:
        armname=selected[family]['arm'];arm=p['arms'][armname]
        checkpoint=args.root/'runs'/arm['phase']/f"PRIVATE-u{arm['checkpoint_updates'][-1]}.pt"
        audit=json.loads((args.root/p['identity']/('audit-'+armname)/'report.json').read_text())
        h=file_sha256(checkpoint)
        if {r['checkpoint_sha256'] for r in audit['rows']}!={h}:
            raise AssertionError('Audited model changed')
        hashes[family]=h
        model=ChunkedPersistentWorkspace(ModelConfig(**p['model_config'])).double().eval()
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(saved['model']);del saved
        steps=int(family!='ordinary');mode='pause' if family=='constant' else 'latent'
        with torch.inference_mode():
            for domain in range(3):
                for length in (129,512):
                    guard(args.root,p);i=domain*128+rank*2
                    x=torch.from_numpy(np.array(tokens[i:i+1,:length],dtype=np.int64))
                    a=model.cached(x,128,steps,mode);b=model.full_reference(x,128,steps,mode)
                    error=float((a.log_softmax(-1)-b.log_softmax(-1)).abs().max())
                    checks.append(dict(family=family,index=i,length=length,maximum_log_probability_error=error))
                    if error>1e-8:raise AssertionError('Mature cache/full-prefix equivalence failed')
                    del a,b
            for indices in indices_for_worker(rank,workers):
                guard(args.root,p)
                x=torch.from_numpy(np.array(tokens[indices,:-1],dtype=np.int64))
                y=torch.from_numpy(np.array(tokens[indices,1:],dtype=np.int64))
                logits=model.cached(x,128,steps,mode)
                losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
                for j,i in enumerate(indices):
                    m=meta[i];rows.append(dict(label=family,index=i,id=m['id'],group=m['group'],
                        domain=m['domain'],targets=512,loss_sum=float(losses[j].sum())))
                del logits,losses
                write_state(args.output/f'worker{rank}-progress.json',dict(family=family,rows=len(rows),wall_seconds=time.time()-start))
        del model
    write_state(args.output/f'PRIVATE-worker{rank}-scores.json',rows)
    write_state(args.output/f'worker{rank}-report.json',dict(status='COMPLETE_CHUNKED_PERSISTENT_EXACT_WORKER',
        checks=checks,rows=len(rows),checkpoint_sha256=hashes,wall_seconds=time.time()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        assigned_chips=0,optimizer_updates=0,protected_groups_scored=0))


def run(args):
    p=verify(args.root,args.plan);guard(args.root,p);start=time.time();args.output.mkdir(exist_ok=False)
    jobs,logs=[],[]
    try:
        for rank in range(p['exact_readout_workers']):
            log=(args.output/f'worker{rank}.log').open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.readout_chunked_persistent',
                '--root',str(args.root),'--plan',str(args.plan),'--output',str(args.output),'--worker',str(rank)],
                stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            guard(args.root,p)
            if time.time()-start>p['exact_readout_seconds_limit']:raise TimeoutError('Exact readout cap')
            if any(j.poll() not in (None,0) for j in jobs):raise RuntimeError('Exact worker failed')
            time.sleep(5)
        if any(j.returncode for j in jobs):raise RuntimeError('Exact worker failed')
        files=[args.output/f'PRIVATE-worker{k}-scores.json' for k in range(p['exact_readout_workers'])]
        rows=[r for f in files for r in json.loads(f.read_text())]
        receipts=[json.loads((args.output/f'worker{k}-report.json').read_text()) for k in range(p['exact_readout_workers'])]
        for family in FAMILIES:
            if len({r['checkpoint_sha256'][family] for r in receipts})!=1:
                raise AssertionError('Same audited checkpoint in all workers required')
            if len([r for r in rows if r['label']==family])!=384:
                raise AssertionError('Full panel coverage required')
        if sum(len(r['checks']) for r in receipts)!=72:raise AssertionError('All72 reference checks required')
        selected=json.loads((args.root/p['identity']/'select/report.json').read_text())['selection']
        write_state(args.output/'report.json',dict(status='COMPLETE_CHUNKED_PERSISTENT_EXACT_READOUT',
            selected_arms={f:s['arm'] for f,s in selected.items()},
            scores={f:summary([r for r in rows if r['label']==f]) for f in FAMILIES},
            workers=receipts,private_score_sha256={f.name:file_sha256(f) for f in files},
            assigned_chips=0,optimizer_updates=0,protected_groups_scored=0,wall_seconds=time.time()-start))
    finally:
        for j in jobs:
            if j.poll() is None:j.terminate()
        for j in jobs:
            try:j.wait(timeout=10)
            except subprocess.TimeoutExpired:j.kill();j.wait()
        for log in logs:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',type=int);a=p.parse_args();run(a) if a.worker is None else worker(a)
