"""Fixed exact-sequential structural development diagnostic, zero updates."""
import argparse,json,resource,shutil,time
from pathlib import Path

ID='interleaved-structural-dev-v1'
BASE='interleaved-lr-tuning-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/structural-dev-v1')
LABELS=('latent','ordinary','pause')


def statistics(rows,replicates=2000,seed=10013):
    import numpy as np
    cells={}
    for r in rows:
        key=(r['label'],r['condition'],r['kind']);cell=cells.setdefault(key,{})
        if r['targets']!=1 or r['group'] in cell:raise AssertionError('Exactly one authored target per kind/group')
        cell[r['group']]=float(r['loss_sum'])
    expected={(m,c,'structural_'+k) for m in LABELS for c in ('full','last32') for k in ('long','short')}
    if set(cells)!=expected:raise AssertionError('Complete model/context/kind grid required')
    groups=sorted(next(iter(cells.values())))
    if not groups or any(set(c)!=set(groups) for c in cells.values()):raise AssertionError('Paired group identity')
    values={k:np.array([v[g] for g in groups]) for k,v in cells.items()}
    if any(not np.isfinite(v).all() for v in values.values()):raise AssertionError('Finite losses required')
    indices=np.random.default_rng(seed).integers(len(groups),size=(replicates,len(groups)))
    def estimate(v):return dict(gain=float(v.mean()),conditional_95_interval=np.quantile(v[indices].mean(1),[.025,.975]).tolist())
    v=lambda m,c,k:values[(m,c,'structural_'+k)]
    means={m:{c:{k:float(v(m,c,k).mean()) for k in ('long','short')} for c in ('full','last32')} for m in LABELS}
    comparisons={}
    for m in ('ordinary','pause'):
        long=v(m,'full','long')-v('latent','full','long');short=v(m,'full','short')-v('latent','full','short')
        comparisons['long_vs_'+m]=estimate(long);comparisons['short_vs_'+m]=estimate(short)
        comparisons['long_minus_short_vs_'+m]=estimate(long-short)
        comparisons['extra_context_utility_vs_'+m]=estimate(v('latent','last32','long')-v('latent','full','long')-v(m,'last32','long')+v(m,'full','long'))
    for m in LABELS:
        for k in ('long','short'):comparisons['full_context_utility_'+m+'_'+k]=estimate(v(m,'last32',k)-v(m,'full',k))
    gates={m:comparisons['long_vs_'+m]['gain']>=.03 and comparisons['long_vs_'+m]['conditional_95_interval'][0]>0 for m in ('ordinary','pause')}
    return dict(means=means,comparisons=comparisons,absolute_long_target_gates=gates,source_groups=len(groups))


def write(path,value):
    from compute_trainium.queue import write_state
    write_state(path,value)


def guard(root,p):
    if time.time()>p['absolute_deadline_unix'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3:
        raise RuntimeError('Frozen CPU time/RSS cap')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<512*1024**3 or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:
        raise RuntimeError('Protected host/disk reserve')


def prepare(root):
    from compute_trainium.training import file_sha256
    prior=Path('results/interleaved_workspace_20261029/tuning-v1/analysis.json')
    selection=json.loads(prior.read_text())['selection'];tuning_path=root/BASE/'plan.json';tuning=json.loads(tuning_path.read_text())
    panel=root/'probability-structural-panel-v1';receipt=json.loads((panel/'report.json').read_text())
    if receipt['status']!='FROZEN_STRUCTURAL_DEPENDENCY_PANEL' or receipt['paired_source_groups']!=141:raise AssertionError('Frozen complete structural panel')
    if any(file_sha256(panel/n)!=h for n,h in receipt['files'].items()):raise AssertionError('Panel bytes changed')
    source_plan=json.loads((root/tuning['data_relative']/'plan.json').read_text())
    if source_plan['excluded_group_files']['probability-structural-panel-v1/PRIVATE-metadata.json']!=receipt['files']['PRIVATE-metadata.json']:
        raise AssertionError('Training must exclude this exact source group panel')
    points={}
    for label in LABELS:
        name=selection[label]['arm']
        if name!=label+'-lr1':raise AssertionError('Previously selected checkpoint required')
        a=tuning['arms'][name];update=a['checkpoint_updates'][-1];checkpoint=root/'runs'/a['phase']/f'PRIVATE-u{update}.pt'
        audit=root/BASE/('audit-'+name)/'report.json';r=json.loads(audit.read_text());h=file_sha256(checkpoint)
        if r['status']!='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS' or {v['checkpoint_sha256'] for v in r['rows'] if v['update']==update}!={h}:
            raise AssertionError('Complete mature audited weights required')
        points[label]=dict(checkpoint_relative=str(checkpoint.relative_to(root)),checkpoint_sha256=h,audit_sha256=file_sha256(audit))
    files=[f for f in tuning['source_sha256'] if f.startswith('compute_trainium/')]
    files+=['scripts/readout_interleaved_structural.py','scripts/evaluate_structural_probability.py','docs/INTERLEAVED_STRUCTURAL_READOUT_20261029.md','tests/test_interleaved_structural_readout.py']
    now=time.time();p=dict(identity=ID,registered_unix=now,absolute_deadline_unix=now+3600,
        tuning_plan_sha256=file_sha256(tuning_path),prior_selection_sha256=file_sha256(prior),points=points,
        panel_relative=panel.name,panel_report_sha256=file_sha256(panel/'report.json'),panel_files=receipt['files'],
        source_sha256={f:file_sha256(f) for f in files},source_groups=141,bootstrap_seed=10013,bootstrap_replicates=2000,
        assigned_chips=0,optimizer_updates=0,protected_groups_scored=0,
        scope='Single-seed source-separated but reused structural DEV. Exact sequential generated states versus equal-data controls. No efficiency, general reasoning, allocation, confirmation or automatic promotion claim.')
    guard(root,p);(root/ID).mkdir();PUBLIC.mkdir(parents=True);write(root/ID/'plan.json',p);write(PUBLIC/'plan.json',p)


def run(root):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.interleaved_sequential_reference import cached_sequential
    from compute_trainium.training import file_sha256
    from scripts.evaluate_structural_probability import truncated_arrays
    out=root/ID;p=json.loads((out/'plan.json').read_text());start=time.time();rows=[]
    result=dict(status='RUNNING_INTERLEAVED_STRUCTURAL_READOUT',assigned_chips=0,optimizer_updates=0,protected_groups_scored=0)
    try:
        if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Frozen source changed')
        panel=root/p['panel_relative'];tuning_path=root/BASE/'plan.json'
        if file_sha256(tuning_path)!=p['tuning_plan_sha256'] or file_sha256(panel/'report.json')!=p['panel_report_sha256']:raise AssertionError('Frozen plans changed')
        if any(file_sha256(panel/n)!=h for n,h in p['panel_files'].items()):raise AssertionError('Frozen source bytes changed')
        torch.set_num_threads(8);torch.set_num_interop_threads(1);tuning=json.loads(tuning_path.read_text());model=make_model(tuning).double().eval()
        meta=json.loads((panel/'PRIVATE-metadata.json').read_text());xs=np.load(panel/'inputs.npy');ys=np.load(panel/'targets.npy');ms=np.load(panel/'mask.npy')
        panels={'full':(xs,ys,ms),'last32':truncated_arrays(xs,ys,ms,meta)};checks=[]
        with torch.inference_mode():
            for label,point in p['points'].items():
                guard(root,p);checkpoint=root/point['checkpoint_relative']
                if file_sha256(checkpoint)!=point['checkpoint_sha256']:raise AssertionError('Audited weights changed')
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(saved['model']);del saved
                if label=='latent':
                    for i in (0,1):
                        x=torch.from_numpy(np.array(panels['last32'][0][i:i+1,:32],dtype=np.int64))
                        a=cached_sequential(model,x).log_softmax(-1);b=model.sequential(x).log_softmax(-1)
                        error=float((a-b).abs().max());checks.append(dict(index=i,maximum_log_probability_error=error))
                        if error>1e-8:raise AssertionError('Sequential FP64 parity failed')
                        del a,b
                for condition,(xx,yy,mm) in panels.items():
                    for i,item in enumerate(meta):
                        guard(root,p);positions=np.flatnonzero(mm[i])
                        if len(positions)!=1:raise AssertionError('One original authored target required')
                        position=int(positions[0]);x=torch.from_numpy(np.array(xx[i:i+1,:position+1],dtype=np.int64));y=torch.tensor([int(yy[i,position])])
                        logits=cached_sequential(model,x) if label=='latent' else model(x,0,label)
                        loss=float(F.cross_entropy(logits[:,-1],y));del logits
                        rows.append(dict(label=label,condition=condition,id=item['id'],group=item['group'],kind=item['kind'],targets=1,loss_sum=loss))
                        if i%16==0:write(out/'progress.json',dict(label=label,condition=condition,completed_cases=i+1,wall_seconds=time.time()-start))
                    write(out/'PRIVATE-scores.json',rows)
        analysis=statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed'])
        if analysis['source_groups']!=p['source_groups']:raise AssertionError('Complete frozen source panel')
        result.update(status='COMPLETE_INTERLEAVED_STRUCTURAL_READOUT',**analysis,checks=checks,trained_seeds=1,automatic_promotion=False,scope=p['scope'])
    except BaseException as e:
        result.update(status='STOPPED_INTERLEAVED_STRUCTURAL_READOUT',error_type=type(e).__name__,error=str(e));raise
    finally:
        result.update(wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(out/'report.json',result);write(PUBLIC/'report.json',result)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--action',choices=['prepare','run'],required=True)
    p=a.parse_args();(prepare if p.action=='prepare' else run)(p.root)
