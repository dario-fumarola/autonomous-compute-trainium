"""Exact CPU readout for frozen replication endpoints; existing DEV only."""
import argparse,json,os,resource,shutil,subprocess,sys,time
from pathlib import Path
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def guard(root,p):
    if time.time()>p['absolute_deadline_unix']:raise RuntimeError('Immutable replication deadline')
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable']<512*1024**3 or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:
        raise RuntimeError('Protected host/disk reserve')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3:raise RuntimeError('4GiB CPU worker bound')


def worker(root,plan,seed,rank):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.tied_interleaved_ordinary import make_model as tied_model
    from compute_trainium.interleaved_sequential_reference import cached_sequential
    from scripts.compare_interleaved_tied import validate_rows
    p=json.loads(plan.read_text());guard(root,p);start=time.time()
    if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Frozen readout source changed')
    if seed not in p['paired_seeds'] or rank not in (0,1):raise ValueError('Frozen seed and worker')
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    evaluation=root/p['evaluation_relative'];files=p['evaluation_files']
    if any(file_sha256(evaluation/f)!=h for f,h in files.items()):raise AssertionError('Existing DEV bytes changed')
    tokens=np.load(evaluation/'broad-tokens.npy',mmap_mode='r');meta=json.loads((evaluation/'PRIVATE-broad-metadata.json').read_text())
    if len(tokens)!=384 or len(meta)!=384:raise AssertionError('Unchanged384-window panel required')
    out=root/p['identity']/f'readout-{seed}';out.mkdir(exist_ok=True);rows=[];checks=[]
    with torch.inference_mode():
        for role in ('latent','ordinary','pause'):
            trial=p['trials'][f'{seed}-{role}'];child=root/trial['identity']/'plan.json'
            if file_sha256(child)!=trial['plan_sha256']:raise AssertionError('Frozen fit plan changed')
            cp=json.loads(child.read_text());arm=cp['arms'][role];update=arm['checkpoint_updates'][-1]
            audit_path=root/trial['identity']/('audit-'+role)/'report.json';audit=json.loads(audit_path.read_text())
            if audit['status']!='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':raise AssertionError('Mature audit required')
            checkpoint=root/'runs'/arm['phase']/f'PRIVATE-u{update}.pt'
            if {r['checkpoint_sha256'] for r in audit['rows'] if r['update']==update}!={file_sha256(checkpoint)}:
                raise AssertionError('Audited checkpoint changed')
            model=(tied_model(cp) if trial['tied'] else make_model(cp)).double().eval()
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(saved['model']);del saved
            if role=='latent':
                for i in range(rank,6,2):
                    guard(root,p);index=(i//2)*128+i%2
                    x=torch.from_numpy(np.array(tokens[index:index+1,:32],dtype=np.int64))
                    a=cached_sequential(model,x);b=model.sequential(x)
                    error=float((a.log_softmax(-1)-b.log_softmax(-1)).abs().max())
                    checks.append(dict(index=index,maximum_log_probability_error=error))
                    if error>1e-8:raise AssertionError('Exact sequential cache/reference parity')
            for left in range(rank*2,len(meta),4):
                guard(root,p);indices=list(range(left,min(left+2,len(meta))))
                x=torch.from_numpy(np.array(tokens[indices,:-1],dtype=np.int64));y=torch.from_numpy(np.array(tokens[indices,1:],dtype=np.int64))
                logits=cached_sequential(model,x) if role=='latent' else model(x,0,arm['workspace_mode'])
                losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
                for j,i in enumerate(indices):
                    m=meta[i];rows.append(dict(label=role,index=i,id=m['id'],group=m['group'],domain=m['domain'],targets=512,loss_sum=float(losses[j].sum())))
                del logits,losses
                if left%32<4:write_state(out/f'worker{rank}-progress.json',dict(role=role,next_index=left+4,wall_seconds=time.time()-start))
            del model
    write_state(out/f'PRIVATE-worker{rank}-scores.json',rows)
    write_state(out/f'worker{rank}-report.json',dict(status='COMPLETE_REPLICATION_EXACT_WORKER',checks=checks,
        rows=len(rows),wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        assigned_chips=0,optimizer_updates=0,protected_groups_scored=0))


def run(root,plan,seed):
    p=json.loads(plan.read_text());guard(root,p);out=root/p['identity']/f'readout-{seed}';out.mkdir(exist_ok=False)
    start=time.time();jobs=[];logs=[]
    try:
        for rank in (0,1):
            log=(out/f'worker{rank}.log').open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.readout_interleaved_replication','--root',str(root),
                '--plan',str(plan),'--seed',str(seed),'--worker',str(rank)],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            guard(root,p)
            if time.time()-start>3600:raise TimeoutError('One-hour paired-seed CPU readout cap')
            if any(j.poll() not in (None,0) for j in jobs):raise RuntimeError('Exact CPU worker failed')
            time.sleep(5)
        if any(j.returncode for j in jobs):raise RuntimeError('Exact CPU worker failed')
        from scripts.compare_interleaved_tied import compare
        rows=[r for rank in (0,1) for r in json.loads((out/f'PRIVATE-worker{rank}-scores.json').read_text())]
        candidate=[r for r in rows if r['label']=='latent']
        result=dict(status='COMPLETE_REPLICATION_EXACT_READOUT',seed=seed,
            comparisons={role:compare(candidate,[r for r in rows if r['label']==role],seed=p['bootstrap_seed']) for role in ('ordinary','pause')},
            workers=[json.loads((out/f'worker{rank}-report.json').read_text()) for rank in (0,1)],
            assigned_chips=0,optimizer_updates=0,protected_groups_scored=0,wall_seconds=time.time()-start)
        write_state(out/'report.json',result)
    finally:
        for j in jobs:
            if j.poll() is None:j.terminate()
        for j in jobs:
            try:j.wait(timeout=10)
            except subprocess.TimeoutExpired:j.kill();j.wait()
        for log in logs:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--worker',type=int);a=p.parse_args()
    run(a.root,a.plan,a.seed) if a.worker is None else worker(a.root,a.plan,a.seed,a.worker)
