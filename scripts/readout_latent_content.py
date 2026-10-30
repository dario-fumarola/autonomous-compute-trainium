"""Bounded, CPU-only path attribution on the frozen billion-target endpoint."""
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

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from compute_trainium.latent_content_intervention import MODES

ID = 'latent-exposure-content-diagnostic-v1'
PARENT = 'latent-learning-exposure-v1'
SOURCE = 'latent-exposure-source-audit-v1'
PUBLIC = Path('results/latent_exposure_20261030/content-diagnostic-v1')


def read(p):
    return json.loads(p.read_text())


def guard(root, plan):
    available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if (time.time() > plan['absolute_deadline_unix'] or available < 512*1024**3 or
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > plan['maximum_worker_rss_bytes'] or
            min(shutil.disk_usage(root).free, shutil.disk_usage('/').free) < 35*1024**3 or
            min(os.statvfs(root).f_favail, os.statvfs('/').f_favail) < 10100):
        raise RuntimeError('Frozen content-readout time/memory/storage bound')


def prepare(root):
    from scripts.readout_interleaved_bracket_type import checkpoint
    parent = read(root/PARENT/'plan.json')
    result = read(root/PARENT/'report.json')
    if result.get('equal_data_development_gate') is not True:
        raise AssertionError('Positive completed equal-data gate required')
    point = checkpoint(root, parent['trials']['latent']['identity'], 'latent')
    files = {str(p.relative_to(root)): file_sha256(p) for p in
             [root/PARENT/'plan.json', root/PARENT/'report.json', root/SOURCE/'report.json', root/SOURCE/'PRIVATE-panel.json']}
    for name, digest in parent['evaluation_files'].items():
        f = root/parent['evaluation_relative']/name
        if file_sha256(f) != digest:
            raise AssertionError('Original broad data changed')
        files[str(f.relative_to(root))] = digest
    for stage in ('broad', 'arithmetic'):
        folder = root/(PARENT+'-equal-data-'+stage)
        if stage == 'broad':
            folder = folder/'readout-730271'
        for f in folder.glob('PRIVATE-*-scores.json'):
            files[str(f.relative_to(root))] = file_sha256(f)
    if files[SOURCE+'/PRIVATE-panel.json'] != parent['panel_sha256']:
        raise AssertionError('Frozen arithmetic subset changed')
    sources = [str(f) for f in Path('compute_trainium').glob('*.py')]
    sources += ['scripts/readout_latent_content.py','scripts/compare_interleaved_tied.py',
                'scripts/analyze_probability_screen.py','scripts/readout_interleaved_bracket_type.py',
                'tests/test_latent_content_intervention.py','tests/test_latent_content_statistics.py',
                'docs/LATENT_CONTENT_DIAGNOSTIC_20261030.md']
    now = time.time()
    plan = dict(identity=ID, registered_unix=now, absolute_deadline_unix=now+3600,
                point=point, frozen_artifacts=files,
                source_sha256={f:file_sha256(f) for f in sources},
                evaluation_relative=parent['evaluation_relative'], modes=list(MODES),
                threads=2, batch_broad=2, batch_arithmetic=8, workers=4,
                maximum_worker_rss_bytes=6*1024**3, bootstrap_seed=40537, bootstrap_replicates=2000,
                assigned_chips=0, optimizer_updates=0, protected_groups_scored=0, terminal_alpha_spent=0,
                scope=Path('docs/LATENT_CONTENT_DIAGNOSTIC_20261030.md').read_text())
    guard(root, plan)
    (root/ID).mkdir(); PUBLIC.mkdir(parents=True)
    write_state(root/ID/'plan.json', plan); write_state(PUBLIC/'plan.json', plan)
    print(json.dumps({'status':'REGISTERED_CPU_LATENT_CONTENT_DIAGNOSTIC','conditions':list(MODES),'seconds_limit':3600}), flush=True)


def verify(root, plan):
    for f, h in plan['source_sha256'].items():
        if file_sha256(f) != h:
            raise AssertionError('Frozen diagnostic code changed')
    for f, h in plan['frozen_artifacts'].items():
        if file_sha256(root/f) != h:
            raise AssertionError('Frozen input or comparator changed')
    point = plan['point']
    if (file_sha256(root/point['plan_relative']) != point['plan_sha256'] or
            file_sha256(root/point['checkpoint_relative']) != point['checkpoint_sha256']):
        raise AssertionError('Audited checkpoint or fit plan changed')


def worker(root, mode):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.latent_content_intervention import cached_intervention, recomputed_intervention
    out=root/ID; plan=read(out/'plan.json'); start=time.time(); cpu_start=time.process_time()
    guard(root, plan); verify(root, plan)
    if mode not in plan['modes']:
        raise ValueError('Unregistered condition')
    os.nice(10)
    torch.set_num_threads(plan['threads']); torch.set_num_interop_threads(1)
    point=plan['point']; fit=read(root/point['plan_relative']); model=make_model(fit).double().eval()
    saved=torch.load(root/point['checkpoint_relative'], map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(saved['model']); del saved
    panel=read(root/SOURCE/'PRIVATE-panel.json'); checks=[]; broad=[]; arithmetic=[]
    with torch.inference_mode():
        for row in panel[:3]:
            x=torch.tensor([row['tokens'][:min(16,len(row['tokens'])-1)]], dtype=torch.long)
            positions=sorted(set((0,x.shape[1]//2,x.shape[1]-1)))
            a=cached_intervention(model,x,mode=mode,positions=positions).log_softmax(-1)
            b=recomputed_intervention(model,x,mode=mode,positions=positions).log_softmax(-1)
            c=cached_intervention(model,x.repeat(2,1),mode=mode,positions=positions).log_softmax(-1)
            errors={'cached_recomputed_error':float((a-b).abs().max()),'batch_individual_error':float((c-a).abs().max())}
            if max(errors.values()) > 1e-8:
                raise AssertionError('Mature-checkpoint intervention parity failed')
            checks.append(errors)
        write_state(out/(mode+'-parity.json'), checks)
        while not all((out/(m+'-parity.json')).exists() for m in plan['modes']):
            guard(root,plan); time.sleep(1)
        folder=root/plan['evaluation_relative']; tokens=np.load(folder/'broad-tokens.npy',mmap_mode='r')
        metadata=read(folder/'PRIVATE-broad-metadata.json')
        for left in range(0,len(metadata),plan['batch_broad']):
            guard(root,plan); indices=list(range(left,min(left+plan['batch_broad'],len(metadata))))
            x=torch.from_numpy(np.array(tokens[indices,:-1],dtype=np.int64))
            y=torch.from_numpy(np.array(tokens[indices,1:],dtype=np.int64))
            logits=cached_intervention(model,x,mode=mode)
            losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
            for j,i in enumerate(indices):
                item=metadata[i]; broad.append(dict(label=mode,index=i,id=item['id'],group=item['group'],
                    domain=item['domain'],targets=512,loss_sum=float(losses[j].sum())))
            del logits,losses
            write_state(out/(mode+'-progress.json'),dict(stage='broad',complete=len(broad),total=len(metadata),wall_seconds=time.time()-start))
        write_state(out/('PRIVATE-'+mode+'-broad.json'),broad)
        groups=defaultdict(list)
        for row in panel: groups[len(row['tokens'])-1].append(row)
        for _,items in sorted(groups.items()):
            for left in range(0,len(items),plan['batch_arithmetic']):
                guard(root,plan); batch=items[left:left+plan['batch_arithmetic']]
                positions=sorted({j-1 for row in batch for j in range(row['answer_start'],len(row['tokens']))})
                x=torch.tensor([row['tokens'][:-1] for row in batch],dtype=torch.long)
                logits=cached_intervention(model,x,mode=mode,positions=positions)
                predictions=logits.argmax(-1); log_probs=logits.log_softmax(-1); where={p:i for i,p in enumerate(positions)}
                for j,row in enumerate(batch):
                    used=[where[p-1] for p in range(row['answer_start'],len(row['tokens']))]
                    target=torch.tensor(row['tokens'][row['answer_start']:],dtype=torch.long)
                    losses=-log_probs[j,used].gather(1,target[:,None]).squeeze(1); correct=predictions[j,used].eq(target)
                    arithmetic.append(dict(index=row['index'],group=row['group'],stratum=row['stratum'],targets=len(used),
                        answer_nll=float(losses.mean()),first_nll=float(losses[0]),exact_prefix_correct=float(correct.all()),
                        loss_sum=float(losses.sum()),correct_tokens=int(correct.sum())))
                del logits,log_probs,predictions
                write_state(out/(mode+'-progress.json'),dict(stage='arithmetic',complete=len(arithmetic),total=len(panel),wall_seconds=time.time()-start))
    write_state(out/('PRIVATE-'+mode+'-arithmetic.json'),arithmetic)
    write_state(out/(mode+'-report.json'),dict(status='COMPLETE_LATENT_CONTENT_WORKER',checks=checks,
        broad_rows=len(broad),arithmetic_rows=len(arithmetic),wall_seconds=time.time()-start,
        process_cpu_seconds=time.process_time()-cpu_start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        checkpoint_sha256=point['checkpoint_sha256']))


def arithmetic_statistics(cells, *, seed, replicates):
    import numpy as np
    reference={r['index']:r for r in cells['authentic']}; bins=('2','3','4-6')
    if not reference or len(reference)!=len(cells['authentic']) or len({r['group'] for r in reference.values()})!=len(reference):
        raise AssertionError('Unique paired source cases required')
    maps={m:{r['index']:r for r in cells[m]} for m in MODES}
    for m in MODES:
        if len(maps[m])!=len(cells[m]) or maps[m].keys()!=reference.keys():
            raise AssertionError('Complete four-condition grid required')
        for i,row in maps[m].items():
            if any(row[k]!=reference[i][k] for k in ('group','stratum','targets')):
                raise AssertionError('Source/target pairing changed')
            if not all(math.isfinite(row[k]) for k in ('answer_nll','first_nll','exact_prefix_correct')):
                raise AssertionError('Finite metrics required')
    indices={b:sorted(i for i,r in reference.items() if r['stratum']==b) for b in bins}
    if any(not x for x in indices.values()): raise AssertionError('Every registered bin required')
    rng=np.random.default_rng(seed); draws={b:rng.integers(len(ids),size=(replicates,len(ids))) for b,ids in indices.items()}
    def values(mode, field): return {b:np.array([maps[mode][i][field] for i in ids]) for b,ids in indices.items()}
    def estimate(v):
        samples=sum(v[b][draws[b]].mean(1) for b in bins)/3
        return dict(mean=float(sum(v[b].mean() for b in bins)/3),bins={b:float(v[b].mean()) for b in bins},
                    conditional_95_interval=np.quantile(samples,[.025,.975]).tolist())
    means={m:{f:estimate(values(m,f)) for f in ('answer_nll','first_nll','exact_prefix_correct')} for m in MODES}
    penalties={}
    for m in MODES[1:]:
        penalties[m]={}
        for f in ('answer_nll','first_nll'):
            a,b=values(m,f),values('authentic',f)
            penalties[m][f]=estimate({k:a[k]-b[k] for k in bins})
    return dict(cases=len(reference),bins={b:len(v) for b,v in indices.items()},means=means,removal_penalties=penalties,
                exact_answer_counts={m:int(sum(r['exact_prefix_correct'] for r in cells[m])) for m in MODES})


def run(root, worker_module='scripts.readout_latent_content'):
    from scripts.compare_interleaved_tied import compare
    from compute_trainium.execution_isolation import managed_running
    out=root/ID; plan=read(out/'plan.json'); start=time.time(); guard(root,plan); verify(root,plan)
    if any(out.glob('*-report.json')): raise FileExistsError('Do not overwrite an existing attempt')
    native_count_at_start=len(managed_running())
    result=dict(status='STOPPED_LATENT_CONTENT_DIAGNOSTIC'); jobs=[]; logs=[]
    try:
        for mode in plan['modes']:
            log=(out/(mode+'.log')).open('w'); logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m',worker_module,'--root',str(root),
                '--action','worker','--mode',mode],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            guard(root,plan)
            if any(j.poll() not in (None,0) for j in jobs): raise RuntimeError('Content-readout worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs): raise RuntimeError('Content-readout worker failed')
        workers={m:read(out/(m+'-report.json')) for m in MODES}
        if any(x['status']!='COMPLETE_LATENT_CONTENT_WORKER' or x['broad_rows']!=384 or x['arithmetic_rows']!=259 or len(x['checks'])!=3 for x in workers.values()):
            raise AssertionError('Complete four-condition result required')
        broad={m:read(out/('PRIVATE-'+m+'-broad.json')) for m in MODES}
        arithmetic={m:read(out/('PRIVATE-'+m+'-arithmetic.json')) for m in MODES}
        old_broad=[r for f in sorted((root/(PARENT+'-equal-data-broad')/'readout-730271').glob('PRIVATE-worker*-scores.json')) for r in read(f) if r['label']=='latent']
        old_arithmetic=read(root/(PARENT+'-equal-data-arithmetic')/'PRIVATE-latent-scores.json')
        broad_ref={r['index']:r for r in old_broad}; arithmetic_ref={r['index']:r for r in old_arithmetic}
        errors=[]
        for row in broad['authentic']:
            ref=broad_ref[row['index']]
            if any(row[k]!=ref[k] for k in ('id','group','domain','targets')): raise AssertionError('Authentic broad pairing changed')
            errors.append(abs(row['loss_sum']-ref['loss_sum'])/row['targets'])
        for row in arithmetic['authentic']:
            ref=arithmetic_ref[row['index']]
            if any(row[k]!=ref[k] for k in ('group','stratum','targets')): raise AssertionError('Authentic arithmetic pairing changed')
            errors.extend(abs(row[k]-ref[k]) for k in ('answer_nll','first_nll','exact_prefix_correct'))
        if max(errors)>1e-8: raise AssertionError('Authentic endpoint failed saved-score replication')
        result.update(status='COMPLETE_LATENT_CONTENT_DIAGNOSTIC',workers=workers,
            authentic_saved_score_maximum_error=max(errors),
            broad={m:compare(broad['authentic'],broad[m],seed=plan['bootstrap_seed'],replicates=plan['bootstrap_replicates']) for m in MODES[1:]},
            arithmetic=arithmetic_statistics(arithmetic,seed=plan['bootstrap_seed'],replicates=plan['bootstrap_replicates']),
            private_score_sha256={f.name:file_sha256(f) for f in out.glob('PRIVATE-*.json')})
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error)); raise
    finally:
        for job in jobs:
            if job.poll() is None: job.terminate()
        for job in jobs:
            try: job.wait(timeout=10)
            except subprocess.TimeoutExpired: job.kill(); job.wait()
        for log in logs: log.close()
        result.update(wall_seconds=time.time()-start,assigned_chips=0,optimizer_updates=0,protected_groups_scored=0,
            terminal_alpha_spent=0,automatic_promotion=False,
            native_owner_count_at_start=native_count_at_start,
            native_owner_count_at_end=len(managed_running()),
            overlapped_existing_native_training=native_count_at_start>0,
            reservation_cost_note='Shared reserved instance; CPU diagnostic overlaps existing Native fit, so do not double-add opportunity cost.',
            scope='Exploratory path dependence at one fixed endpoint, not causal reasoning or an efficiency claim.',plan_sha256=file_sha256(out/'plan.json'))
        write_state(out/'report.json',result); write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--action',choices=('prepare','run','worker'),required=True); parser.add_argument('--mode',choices=MODES)
    args=parser.parse_args()
    if args.action=='worker': worker(args.root,args.mode)
    else: (prepare if args.action=='prepare' else run)(args.root)
