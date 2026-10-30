"""Existing authored code targets after the fixed billion-target learning fit."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.readout_direct_arithmetic import guard
from scripts.readout_interleaved_bracket_type import checkpoint,typed_statistics

ID='latent-learning-code-readout-v1'
PUBLIC=Path('results/latent_exposure_20261030/code-readout-v1')
ROLES=('latent','ordinary','pause')


def read(path):return json.loads(Path(path).read_text())


def last_logits(model,tokens,role):
    import torch
    from torch.nn import functional as F
    from compute_trainium.interleaved_last_readout import cached_last_logits
    if role=='latent':return cached_last_logits(model,tokens)
    source=model.embeddings(tokens);pos=torch.arange(tokens.shape[1],device=tokens.device)
    if role=='ordinary':h=model.body_at(source,pos)
    elif role=='pause':
        constant=model.source_scale*model.embedding.weight.mean(0).expand_as(source)
        h=model.body_at(model.interleave(source,constant),pos.repeat_interleave(2))
    else:raise ValueError('Only registered computation paths')
    return F.linear(h[:,-1],model.prediction_weight)


def prepare(root):
    old=root/'interleaved-bracket-type-dev-v1';p=read(old/'plan.json')
    prior=read(old/'report.json')
    if prior['status']!='COMPLETE_BRACKET_TYPE_DIAGNOSTIC':raise AssertionError('Original complete diagnostic required')
    p.pop('pending_tied_plan_sha256',None);p.pop('pending_control',None)
    points={r:checkpoint(root,'latent-learning-exposure-v1-'+r,r) for r in ROLES}
    for role,point in points.items():
        fit=read(root/point['plan_relative'])
        if fit['seed']!=730271 or fit['arms'][role]['real_targets']!=1000060368:
            raise AssertionError('Exactly the registered billion-target endpoints')
        source_plan=read(root/fit['data_relative']/'plan.json')
        if source_plan['excluded_group_files']['probability-structural-panel-v1/PRIVATE-metadata.json']!=p['panel_files']['PRIVATE-metadata.json']:
            raise AssertionError('Code evaluation source groups excluded from TRAIN')
    sources={str(f):file_sha256(f) for f in Path('compute_trainium').glob('*.py')}
    for f in ('scripts/readout_latent_learning_code.py','scripts/readout_interleaved_bracket_type.py',
              'scripts/readout_direct_arithmetic.py','scripts/readout_interleaved_structural.py',
              'scripts/evaluate_structural_probability.py','tests/test_latent_learning_code.py',
              'docs/LATENT_LEARNING_CODE_READOUT_20261030.md'):
        sources[f]=file_sha256(f)
    now=time.time();p.update(identity=ID,points=points,registered_unix=now,absolute_deadline_unix=now+1800,
        threads=8,batch=8,maximum_worker_rss_bytes=6*1024**3,source_sha256=sources,
        old_result_sha256=file_sha256(old/'report.json'),classes_relative=str((old/'PRIVATE-classes.json').relative_to(root)),
        assigned_chips=0,optimizer_updates=0,protected_groups_scored=0,terminal_alpha_spent=0,
        concurrent_native_owner='latent-exposure-work-recovery-v1',automatic_promotion=False,
        scope=Path('docs/LATENT_LEARNING_CODE_READOUT_20261030.md').read_text())
    out=root/ID
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every readout attempt')
    guard(root,p);out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def worker(root,role):
    import numpy as np
    import torch
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.tied_interleaved_ordinary import make_model as tied_model
    from scripts.evaluate_structural_probability import truncated_arrays
    p=read(root/ID/'plan.json');point=p['points'][role];start=time.time();guard(root,p)
    if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Frozen readout source changed')
    panel=root/p['panel_relative']
    if file_sha256(panel/'report.json')!=p['panel_report_sha256'] or any(file_sha256(panel/f)!=h for f,h in p['panel_files'].items()):
        raise AssertionError('Original code panel changed')
    if file_sha256(root/p['classes_relative'])!=p['classes_sha256']:raise AssertionError('Original type classes changed')
    if file_sha256(root/point['plan_relative'])!=point['plan_sha256'] or file_sha256(root/point['checkpoint_relative'])!=point['checkpoint_sha256']:
        raise AssertionError('Audited endpoint changed')
    torch.set_num_threads(p['threads']);torch.set_num_interop_threads(1)
    fit=read(root/point['plan_relative']);model=(tied_model(fit) if role=='ordinary' else make_model(fit)).double().eval()
    saved=torch.load(root/point['checkpoint_relative'],map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(saved['model']);del saved
    xx=np.load(panel/'inputs.npy');yy=np.load(panel/'targets.npy');mm=np.load(panel/'mask.npy');meta=read(panel/'PRIVATE-metadata.json')
    panels={'full':(xx,yy,mm),'last32':truncated_arrays(xx,yy,mm,meta)}
    classes=read(root/p['classes_relative']);types=(')',']','}');ids={k:torch.tensor(classes[k]) for k in types}
    rows=[];checks=[]
    with torch.inference_mode():
        for i in (0,1,2):
            x=torch.from_numpy(np.array(xx[i:i+1,:16],dtype=np.int64))
            a=last_logits(model,x,role).log_softmax(-1)
            b=(model.sequential(x) if role=='latent' else model(x,mode=role))[:,-1].log_softmax(-1)
            c=last_logits(model,x.repeat(2,1),role).log_softmax(-1)
            error=float((a-b).abs().max());batch_error=float((a-c).abs().max())
            if max(error,batch_error)>1e-8:raise AssertionError('Actual-weight last/full and batch parity')
            checks.append(dict(index=i,full_prefix_error=error,batch_error=batch_error))
        for condition,(xs,ys,ms) in panels.items():
            bylength=defaultdict(list)
            for i,item in enumerate(meta):
                pos=np.flatnonzero(ms[i])
                if len(pos)!=1:raise AssertionError('One unchanged natural target')
                bylength[int(pos[0])+1].append(i)
            for length,indices in sorted(bylength.items()):
                for start_index in range(0,len(indices),p['batch']):
                    guard(root,p);batch=indices[start_index:start_index+p['batch']]
                    x=torch.from_numpy(np.array(xs[batch,:length],dtype=np.int64));logits=last_logits(model,x,role)
                    for j,i in enumerate(batch):
                        item=meta[i];target=int(ys[i,length-1]);z=logits[j]
                        if target not in classes[item['closer_type']]:raise AssertionError('Frozen target type mapping')
                        masses=torch.stack([torch.logsumexp(z[ids[k]],0) for k in types]);gold=types.index(item['closer_type'])
                        rows.append(dict(label=role,condition=condition,id=item['id'],group=item['group'],kind=item['kind'],closer=item['closer_type'],
                            targets=1,loss_sum=float(torch.logsumexp(z,0)-z[target]),
                            type_nll=float(torch.logsumexp(masses,0)-masses[gold]),type_correct=int(int(masses.argmax())==gold)))
                    write_state(root/ID/(role+'-progress.json'),dict(condition=condition,rows=len(rows),wall_seconds=time.time()-start))
    write_state(root/ID/('PRIVATE-'+role+'-scores.json'),rows)
    write_state(root/ID/(role+'-report.json'),dict(status='COMPLETE_LEARNING_CODE_WORKER',rows=len(rows),checks=checks,
        wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def run(root):
    from scripts.readout_interleaved_structural import statistics
    p=read(root/ID/'plan.json');start=time.time();jobs=[];logs=[];result=dict(status='STOPPED_LEARNING_CODE_READOUT')
    try:
        for role in ROLES:
            log=(root/ID/(role+'.log')).open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.readout_latent_learning_code','--root',str(root),'--action','worker','--role',role],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            guard(root,p)
            if any(j.poll() not in (None,0) for j in jobs):raise RuntimeError('Bounded code readout worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs):raise RuntimeError('Bounded code readout worker failed')
        workers={r:read(root/ID/(r+'-report.json')) for r in ROLES}
        if any(v['status']!='COMPLETE_LEARNING_CODE_WORKER' or len(v['checks'])!=3 or v['rows']!=564 for v in workers.values()):
            raise AssertionError('All models, original cases and numerical checks required')
        rows=[v for r in ROLES for v in read(root/ID/('PRIVATE-'+r+'-scores.json'))]
        result.update(status='COMPLETE_LEARNING_CODE_READOUT',typed=typed_statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed']),
            raw=statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed']),workers=workers)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        for j in jobs:
            if j.poll() is None:j.terminate()
        for j in jobs:
            try:j.wait(timeout=10)
            except subprocess.TimeoutExpired:j.kill();j.wait()
        for log in logs:log.close()
        result.update(wall_seconds=time.time()-start,assigned_chips=0,optimizer_updates=0,
            protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False,
            concurrent_native_owner=p['concurrent_native_owner'],plan_sha256=file_sha256(root/ID/'plan.json'))
        write_state(root/ID/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True)
    a.add_argument('--action',choices=('prepare','run','worker'),required=True);a.add_argument('--role',choices=ROLES)
    args=a.parse_args();worker(args.root,args.role) if args.action=='worker' else (prepare if args.action=='prepare' else run)(args.root)
