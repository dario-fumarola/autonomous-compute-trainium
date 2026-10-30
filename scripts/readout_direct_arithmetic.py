"""All saved seeds and controls on a frozen, source-separated arithmetic panel."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.readout_interleaved_bracket_type import checkpoint

ID='direct-arithmetic-readout-v1'
PANEL='direct-arithmetic-panel-v1'
PUBLIC=Path('results/direct_arithmetic_20261030/readout-v1')
SEEDS=(9901,730201,730237)
ROLES=('latent','ordinary','pause')
BINS=('2','3','4-6')


def read(path):return json.loads(path.read_text())


def statistics(rows,replicates=2000,seed=40530):
    import numpy as np
    cells={(s,r):{v['index']:v for v in rows if v['seed']==s and v['role']==r} for s in SEEDS for r in ROLES}
    reference=cells[(SEEDS[0],ROLES[0])]
    expected=set(reference)
    if not expected or len({r['group'] for r in reference.values()})!=len(expected):
        raise AssertionError('Nonempty one-source-group-per-case panel required')
    for values in cells.values():
        if set(values)!=expected:raise AssertionError('Complete nine-cell comparison required')
        for i,v in values.items():
            if any(v[k]!=reference[i][k] for k in ('group','stratum','targets')):
                raise AssertionError('Paired source identities and answers required')
            if not all(math.isfinite(v[k]) for k in ('answer_nll','first_nll','exact_prefix_correct')):
                raise AssertionError('Finite complete metrics required')
    if len(rows)!=9*len(expected):raise AssertionError('No duplicate scoring records')
    indices={b:sorted(i for i in expected if reference[i]['stratum']==b) for b in BINS}
    if any(not v for v in indices.values()):raise AssertionError('Every operation bin required')
    rng=np.random.default_rng(seed)
    draws={b:rng.integers(len(v),size=(replicates,len(v))) for b,v in indices.items()}

    def values(cell,field):return {b:np.array([cell[i][field] for i in indices[b]]) for b in BINS}
    def estimate(v):
        samples=sum(v[b][draws[b]].mean(1) for b in BINS)/3
        return dict(mean=float(sum(v[b].mean() for b in BINS)/3),conditional_95_interval=np.quantile(samples,[.025,.975]).tolist())

    means={};contrasts={};gates={}
    for s in SEEDS:
        means[str(s)]={}
        for role in ROLES:
            cell=cells[(s,role)]
            means[str(s)][role]={field:dict(macro=estimate(values(cell,field)),
                bins={b:float(values(cell,field)[b].mean()) for b in BINS})
                for field in ('answer_nll','first_nll','exact_prefix_correct')}
        contrasts[str(s)]={};gates[str(s)]={}
        for control in ('ordinary','pause'):
            gain={b:values(cells[(s,control)],'answer_nll')[b]-values(cells[(s,'latent')],'answer_nll')[b] for b in BINS}
            e=estimate(gain);e['bins']={b:float(gain[b].mean()) for b in BINS}
            contrasts[str(s)][control]=e
            gates[str(s)][control]=e['mean']>=.03 and e['conditional_95_interval'][0]>0
    across={}
    for control in ('ordinary','pause'):
        x=np.array([contrasts[str(s)][control]['mean'] for s in SEEDS])
        mean=float(x.mean());sd=float(x.std(ddof=1));half=4.3026527299*sd/math.sqrt(3)
        across[control]=dict(mean=mean,paired_seed_sd=sd,nominal_three_seed_95_t_interval=[mean-half,mean+half],
            note='Three existing seeds, including the originally tuned seed; weak seed-variance evidence, not confirmation.')
    return dict(cases=len(expected),groups=len(expected),bins={b:len(v) for b,v in indices.items()},
                means=means,contrasts=contrasts,across_seeds=across,per_seed_lead_gates=gates,
                task_specific_lead=all(v for g in gates.values() for v in g.values()),
                scope='Frozen natural direct-calculation DEV diagnostic. Equal source exposure; latent training work3.782219x ordinary. No efficiency, adaptive allocation or general reasoning claim.')


def prepare(root):
    panel=root/PANEL;receipt=read(panel/'report.json')
    if receipt['status']!='FROZEN_DIRECT_ARITHMETIC_DEVELOPMENT_PANEL' or any(receipt[k] for k in
            ('source_group_overlap','verified_training_expression_overlap','protected_group_overlap','model_scores')):
        raise AssertionError('Completed unscored source-separated panel required')
    points={}
    original={'latent':('interleaved-mixed-round-tuning-v1','mixed-lr1'),
              'ordinary':('interleaved-tied-ordinary-tuning-v1','tied-lr1'),
              'pause':('interleaved-lr-tuning-v1','pause-lr1')}
    for s in SEEDS:
        for role in ROLES:
            base,arm=original[role] if s==9901 else (f'interleaved-seed-replication-v1-s{s}-{role}',role)
            point=checkpoint(root,base,arm);fit=read(root/point['plan_relative'])
            if fit['seed']!=s or fit['data_relative']!='indexed-natural-sources-v1' or fit['arms'][arm]['real_targets']!=128025288:
                raise AssertionError('Fixed audited scientific endpoints at equal real data required')
            point.update(seed=s,role=role);points[f'{s}-{role}']=point
    files=[str(f) for f in Path('compute_trainium').glob('*.py')]
    files+=['scripts/readout_direct_arithmetic.py','scripts/readout_interleaved_bracket_type.py',
            'tests/test_direct_arithmetic_readout.py','docs/DIRECT_ARITHMETIC_READOUT_20261030.md']
    now=time.time();p=dict(identity=ID,registered_unix=now,absolute_deadline_unix=now+1800,
        seconds_limit=1800,threads=4,batch=8,maximum_worker_rss_bytes=6*1024**3,points=points,
        panel_report_sha256=file_sha256(panel/'report.json'),panel_files=receipt['files'],
        source_sha256={f:file_sha256(f) for f in files},bootstrap_seed=40530,bootstrap_replicates=2000,
        assigned_chips=0,optimizer_updates=0,protected_groups_scored=0,terminal_alpha_spent=0,
        scope=Path('docs/DIRECT_ARITHMETIC_READOUT_20261030.md').read_text())
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def guard(root,p):
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if (time.time()>p['absolute_deadline_unix'] or available<512*1024**3
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['maximum_worker_rss_bytes']
            or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3
            or min(os.statvfs(root).f_favail,os.statvfs('/').f_favail)<10100):
        raise RuntimeError('Frozen readout resource bound')


def worker(root,key):
    import torch
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.tied_interleaved_ordinary import make_model as tied_model
    from compute_trainium.interleaved_selected_readout import cached_selected_logits
    p=read(root/ID/'plan.json');point=p['points'][key];started=time.time();guard(root,p)
    if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Frozen readout source changed')
    if file_sha256(root/point['plan_relative'])!=point['plan_sha256'] or file_sha256(root/point['checkpoint_relative'])!=point['checkpoint_sha256']:
        raise AssertionError('Audited endpoint changed')
    if (file_sha256(root/PANEL/'report.json')!=p['panel_report_sha256']
            or any(file_sha256(root/PANEL/f)!=h for f,h in p['panel_files'].items())):
        raise AssertionError('Frozen source panel changed')
    torch.set_num_threads(p['threads']);torch.set_num_interop_threads(1)
    fit=read(root/point['plan_relative']);role=point['role']
    model=(tied_model(fit) if role=='ordinary' else make_model(fit)).double().eval()
    saved=torch.load(root/point['checkpoint_relative'],map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(saved['model']);del saved
    panel=read(root/PANEL/'PRIVATE-panel.json');rows=[];checks=[]

    def score(x,positions):
        if role=='latent':return cached_selected_logits(model,x,positions)
        return model(x,mode=role)[:,positions]

    with torch.inference_mode():
        for row in panel[:3]:
            x=torch.tensor([row['tokens'][:min(16,len(row['tokens'])-1)]],dtype=torch.long)
            positions=sorted(set((0,x.shape[1]//2,x.shape[1]-1)))
            a=score(x,positions).log_softmax(-1)
            b=(model.sequential(x) if role=='latent' else model(x,mode=role)).log_softmax(-1)[:,positions]
            error=float((a-b).abs().max())
            if error>1e-8:raise AssertionError('Selected/full-prefix numerical discrepancy')
            batch=score(x.repeat(2,1),positions).log_softmax(-1)
            batch_error=float((batch-a).abs().max())
            if batch_error>1e-8:raise AssertionError('Batch/individual numerical discrepancy')
            checks.append(dict(selected_full_error=error,batch_individual_error=batch_error))
        groups=defaultdict(list)
        for row in panel:groups[len(row['tokens'])-1].append(row)
        for n,items in sorted(groups.items()):
            for first in range(0,len(items),p['batch']):
                guard(root,p);batch=items[first:first+p['batch']]
                positions=sorted({j-1 for row in batch for j in range(row['answer_start'],len(row['tokens']))})
                x=torch.tensor([row['tokens'][:-1] for row in batch],dtype=torch.long)
                logits=score(x,positions);predicted=logits.argmax(-1)
                log_probs=logits.log_softmax(-1);where={v:i for i,v in enumerate(positions)}
                for j,row in enumerate(batch):
                    used=[where[v-1] for v in range(row['answer_start'],len(row['tokens']))]
                    target=torch.tensor(row['tokens'][row['answer_start']:],dtype=torch.long)
                    loss=-log_probs[j,used].gather(1,target[:,None]).squeeze(1)
                    correct=predicted[j,used].eq(target)
                    rows.append(dict(index=row['index'],group=row['group'],stratum=row['stratum'],seed=point['seed'],role=role,
                        targets=len(used),answer_nll=float(loss.mean()),first_nll=float(loss[0]),
                        exact_prefix_correct=float(correct.all()),loss_sum=float(loss.sum()),correct_tokens=int(correct.sum())))
                del logits,log_probs,predicted
                write_state(root/ID/f'{key}-progress.json',dict(cases=len(rows),total=len(panel),wall_seconds=time.time()-started))
    write_state(root/ID/f'PRIVATE-{key}-scores.json',rows)
    write_state(root/ID/f'{key}-report.json',dict(status='COMPLETE_DIRECT_ARITHMETIC_WORKER',cases=len(rows),checks=checks,
        checkpoint_sha256=point['checkpoint_sha256'],wall_seconds=time.time()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def run(root):
    if managed_running():raise RuntimeError('Native owners must close before CPU readout')
    p=read(root/ID/'plan.json');guard(root,p);started=time.time();jobs=[];logs=[];result=dict(status='STOPPED_DIRECT_ARITHMETIC_READOUT')
    try:
        for key in p['points']:
            log=(root/ID/f'{key}.log').open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.readout_direct_arithmetic','--root',str(root),
                '--action','worker','--key',key],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            guard(root,p)
            if any(j.poll() not in (None,0) for j in jobs):raise RuntimeError('Frozen CPU worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs):raise RuntimeError('Frozen CPU worker failed')
        receipts={key:read(root/ID/f'{key}-report.json') for key in p['points']}
        if any(v['status']!='COMPLETE_DIRECT_ARITHMETIC_WORKER' or len(v['checks'])!=3 for v in receipts.values()):
            raise AssertionError('All nine workers and27 numerical records required')
        rows=[row for key in p['points'] for row in read(root/ID/f'PRIVATE-{key}-scores.json')]
        result.update(status='COMPLETE_DIRECT_ARITHMETIC_READOUT',analysis=statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed']),
            workers=receipts,private_score_sha256={f.name:file_sha256(f) for f in (root/ID).glob('PRIVATE-*-scores.json')})
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        for j in jobs:
            if j.poll() is None:j.terminate()
        for j in jobs:
            try:j.wait(timeout=10)
            except subprocess.TimeoutExpired:j.kill();j.wait()
        for log in logs:log.close()
        result.update(wall_seconds=time.time()-started,assigned_chips=0,optimizer_updates=0,protected_groups_scored=0,
            terminal_alpha_spent=0,automatic_promotion=False,plan_sha256=file_sha256(root/ID/'plan.json'))
        write_state(root/ID/'report.json',result);write_state(PUBLIC/'report.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('analysis','workers','private_score_sha256')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run','worker'],required=True);p.add_argument('--key')
    a=p.parse_args();worker(a.root,a.key) if a.action=='worker' else (prepare if a.action=='prepare' else run)(a.root)
