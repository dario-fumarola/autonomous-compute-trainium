"""Fixed selected-checkpoint CPU sequential readout; no fitting or selection."""
import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

ID = 'interleaved-mixed-sequential-dev-v1'
BASE = 'interleaved-mixed-round-tuning-v1'
PUBLIC = Path('results/interleaved_workspace_20261029/mixed-sequential-dev-v1')


def write(path, value):
    from compute_trainium.queue import write_state
    write_state(path, value)


def guard(root, p):
    if time.time() > p['absolute_deadline_unix']:
        raise RuntimeError('Immutable sequential readout deadline')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > p['worker_rss_limit_bytes']:
        raise RuntimeError('CPU worker RSS limit')
    mem = {x.split(':')[0]: int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable'] < 512*1024**3 or min(shutil.disk_usage(root).free, shutil.disk_usage('/').free) < 35*1024**3:
        raise RuntimeError('Protected host or output-disk reserve')


def prepare(root):
    from compute_trainium.training import file_sha256
    from scripts.run_interleaved_frontier_qualification import mechanism_gate
    gates=mechanism_gate()
    analysis_path=Path('results/interleaved_workspace_20261029/mixed-round-tuning-v1/analysis.json')
    selection=json.loads(analysis_path.read_text())['selection'];selected=selection['arm']
    tuning_path=root/BASE/'plan.json';tuning=json.loads(tuning_path.read_text())
    arm=tuning['arms'][selected];update=arm['checkpoint_updates'][-1]
    checkpoint=root/'runs'/arm['phase']/f'PRIVATE-u{update}.pt'
    audit_path=root/BASE/('audit-'+selected)/'report.json';audit=json.loads(audit_path.read_text())
    if audit['status']!='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':raise AssertionError('Mature audit required')
    checkpoint_sha=file_sha256(checkpoint)
    if {r['checkpoint_sha256'] for r in audit['rows'] if r['update']==update}!={checkpoint_sha}:
        raise AssertionError('Audited checkpoint changed')
    evaluation=root/tuning['evaluation_relative'];data_report=json.loads((evaluation/'report.json').read_text())
    data_files={n:data_report['files'][n] for n in ('broad-tokens.npy','PRIVATE-broad-metadata.json')}
    if any(file_sha256(evaluation/n)!=h for n,h in data_files.items()):raise AssertionError('Frozen DEV changed')
    metadata=json.loads((evaluation/'PRIVATE-broad-metadata.json').read_text())
    if len(metadata)!=384 or len({r['id'] for r in metadata})!=384:raise AssertionError('Complete frozen panel required')
    parity_indices=[i for d in ('web','math','code') for i in [j for j,r in enumerate(metadata) if r['domain']==d][:2]]
    if len(parity_indices)!=6:raise AssertionError('Two checks per domain')
    source_files=[f for f in tuning['source_sha256'] if f.startswith('compute_trainium/')]
    source_files+=['scripts/readout_interleaved_mixed_sequential.py','scripts/analyze_probability_screen.py',
                  'docs/INTERLEAVED_MIXED_SEQUENTIAL_READOUT_20261029.md']
    control_files={}
    for base in ('interleaved-lr-tuning-v1',BASE):
        controls=root/'runs'/(base+'-evaluation')
        control_files.update({str((controls/f'PRIVATE-rank{r}-scores.json').relative_to(root)):
                             file_sha256(controls/f'PRIVATE-rank{r}-scores.json') for r in range(64)})
    now=time.time()
    p=dict(identity=ID,registered_unix=now,absolute_deadline_unix=now+3600,
        workers=2,threads=8,batch=2,worker_rss_limit_bytes=4*1024**3,
        selected_arm=selected,output_label=selected+'-sequential',comparison_labels=['ordinary-lr1-main','pause-lr1-main',selected+'-main',selected+'-supported3'],
        checkpoint_relative=str(checkpoint.relative_to(root)),checkpoint_sha256=checkpoint_sha,
        audit_sha256=file_sha256(audit_path),tuning_plan_sha256=file_sha256(tuning_path),
        prior_selection_sha256=file_sha256(analysis_path),mechanism_gate_sha256=gates,
        evaluation_relative=tuning['evaluation_relative'],data_sha256=data_files,control_sha256=control_files,
        source_sha256={f:file_sha256(f) for f in source_files},parity_indices=parity_indices,parity_targets=32,
        parity_maximum_nll_error=1e-8,bootstrap_seed=10003,bootstrap_replicates=2000,precision='FP64',
        optimizer_updates=0,assigned_chips=0,protected_groups_scored=0,
        scope='Exact sequential deployment readout of the previously K2-selected mixed endpoint. Reused single-seed DEV, no new selection, inference speed, general reasoning, allocation or efficiency claim.')
    guard(root,p);(root/ID).mkdir();PUBLIC.mkdir(parents=True)
    write(root/ID/'plan.json',p);write(PUBLIC/'plan.json',p)
    print(json.dumps(dict(status='REGISTERED_MIXED_SEQUENTIAL_READOUT',selected=selected,workers=2)))


def worker(root, rank):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.interleaved_sequential_reference import cached_sequential
    from compute_trainium.training import file_sha256
    out=root/ID;p=json.loads((out/'plan.json').read_text());guard(root,p)
    torch.set_num_threads(p['threads']);torch.set_num_interop_threads(1)
    tuning=json.loads((root/BASE/'plan.json').read_text())
    model=make_model(tuning).double().eval()
    saved=torch.load(root/p['checkpoint_relative'],map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(saved['model']);del saved
    folder=root/p['evaluation_relative'];tokens=np.load(folder/'broad-tokens.npy',mmap_mode='r')
    metadata=json.loads((folder/'PRIVATE-broad-metadata.json').read_text())
    rows=[];checks=[];start=time.time()
    with torch.inference_mode():
        for index in p['parity_indices'][rank::p['workers']]:
            guard(root,p);n=p['parity_targets'];x=torch.from_numpy(np.array(tokens[index:index+1,:n],dtype=np.int64));y=torch.from_numpy(np.array(tokens[index:index+1,1:n+1],dtype=np.int64))
            a=cached_sequential(model,x);b=model.sequential(x)
            la=F.cross_entropy(a.flatten(0,1),y.flatten(),reduction='none')
            lb=F.cross_entropy(b.flatten(0,1),y.flatten(),reduction='none')
            error=float((la-lb).abs().max())
            checks.append(dict(index=index,maximum_target_loss_error=error))
            if error>p['parity_maximum_nll_error']:raise AssertionError('Independent sequential parity failed')
            del a,b,la,lb
        write(out/f'parity-{rank}.json',checks)
        # Global parity barrier: do not score the full panel if another check fails.
        while not all((out/f'parity-{r}.json').exists() for r in range(p['workers'])):
            guard(root,p);time.sleep(1)
        for left in range(rank*p['batch'],len(tokens),p['workers']*p['batch']):
            guard(root,p);indices=list(range(left,min(left+p['batch'],len(tokens))))
            batch=np.array(tokens[indices],dtype=np.int64);x=torch.from_numpy(batch[:,:-1]);y=torch.from_numpy(batch[:,1:])
            logits=cached_sequential(model,x)
            losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
            for j,index in enumerate(indices):
                item=metadata[index]
                rows.append(dict(label=p['output_label'],panel='broad',index=index,id=item['id'],group=item['group'],
                    domain=item['domain'],kind=item['kind'],targets=512,loss_sum=float(losses[j].sum()),first_loss=float(losses[j,0])))
            del logits,losses,x,y
            write(out/f'PRIVATE-worker{rank}-scores.json',rows)
            write(out/f'progress-worker{rank}.json',dict(completed_views=len(rows),wall_seconds=time.time()-start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    write(out/f'worker{rank}-report.json',dict(status='COMPLETE_SEQUENTIAL_DEV_PARTITION',views=len(rows),checks=checks,
        wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def run(root):
    from compute_trainium.training import file_sha256
    from scripts.analyze_probability_screen import score,paired
    out=root/ID;p=json.loads((out/'plan.json').read_text());processes=[];logs=[];start=time.time()
    result=dict(status='RUNNING_SEQUENTIAL_DEV_READOUT',assigned_chips=0,optimizer_updates=0,protected_groups_scored=0)
    try:
        for f,h in p['source_sha256'].items():
            if file_sha256(f)!=h:raise AssertionError('Frozen source changed: '+f)
        for f,h in p['data_sha256'].items():
            if file_sha256(root/p['evaluation_relative']/f)!=h:raise AssertionError('Frozen development bytes changed')
        for f,h in p['control_sha256'].items():
            if file_sha256(root/f)!=h:raise AssertionError('Frozen comparator bytes changed')
        if file_sha256(root/p['checkpoint_relative'])!=p['checkpoint_sha256'] or file_sha256(root/BASE/'plan.json')!=p['tuning_plan_sha256']:
            raise AssertionError('Frozen model or training plan changed')
        for rank in range(p['workers']):
            log=open(out/f'worker{rank}.log','w');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-m','scripts.readout_interleaved_mixed_sequential','--root',str(root),'--action','worker','--rank',str(rank)],stdout=log,stderr=subprocess.STDOUT))
        while any(job.poll() is None for job in processes):
            guard(root,p)
            if any(job.poll() not in (None,0) for job in processes):raise RuntimeError('Sequential CPU partition failed; inspect preserved worker log')
            time.sleep(2)
        if any(job.returncode!=0 for job in processes):raise RuntimeError('Sequential CPU partition failed')
        rows=[r for rank in range(p['workers']) for r in json.loads((out/f'PRIVATE-worker{rank}-scores.json').read_text())]
        if len(rows)!=384 or {r['index'] for r in rows}!=set(range(384)):raise AssertionError('Every frozen target required exactly once')
        controls=[r for f in p['control_sha256'] for r in json.loads((root/f).read_text())]
        domains={d:score([r for r in rows if r['domain']==d]) for d in ('web','math','code')}
        comparisons={}
        for label in p['comparison_labels']:
            selected=[r for r in controls if r['label']==label and r['panel']=='broad']
            comparisons[label]=dict(broad=paired(selected,rows,seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates'],weighted=True),
                domains={d:paired([r for r in selected if r['domain']==d],[r for r in rows if r['domain']==d],seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates']) for d in domains})
        result.update(status='COMPLETE_SEQUENTIAL_DEV_READOUT',domains=domains,broad_nll=sum(w*domains[d] for d,w in zip(('web','math','code'),(.6,.2,.2))),
            comparisons=comparisons,workers=[json.loads((out/f'worker{rank}-report.json').read_text()) for rank in range(p['workers'])],
            checkpoint_sha256=p['checkpoint_sha256'],scope=p['scope'],automatic_promotion=False,trained_seeds=1)
    except BaseException as error:
        result.update(status='STOPPED_SEQUENTIAL_DEV_READOUT',error_type=type(error).__name__,error=str(error));raise
    finally:
        for job in processes:
            if job.poll() is None:job.terminate()
        for job in processes:
            try:job.wait(timeout=10)
            except subprocess.TimeoutExpired:job.kill();job.wait()
        for log in logs:log.close()
        result['wall_seconds']=time.time()-start;write(out/'report.json',result);write(PUBLIC/'report.json',result)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True)
    a.add_argument('--action',choices=['prepare','run','worker'],required=True);a.add_argument('--rank',type=int)
    args=a.parse_args()
    if args.action=='worker':worker(args.root,args.rank)
    else:(prepare if args.action=='prepare' else run)(args.root)
