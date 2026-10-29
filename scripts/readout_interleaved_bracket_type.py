"""Separate bracket-type selection from punctuation propensity on authentic code."""
import argparse,json,math,resource,shutil,time
from pathlib import Path
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID='interleaved-bracket-type-dev-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/bracket-type-dev-v1')
TIED='interleaved-tied-ordinary-tuning-v1'
LABELS=('latent','pause','ordinary')
TYPES=(')',']','}')


def typed_statistics(rows,replicates=2000,seed=10033):
    import numpy as np
    cells={};classes={}
    for r in rows:
        key=(r['label'],r['condition'],r['kind']);cell=cells.setdefault(key,{})
        if r['group'] in cell or r['closer'] not in TYPES:raise AssertionError('Unique paired typed target required')
        value=(float(r['type_nll']),float(r['type_correct']))
        if not all(math.isfinite(x) for x in value) or value[0]<0 or value[1] not in (0.,1.):raise AssertionError('Finite typed metrics required')
        cell[r['group']]=value
        if classes.setdefault(r['group'],r['closer'])!=r['closer']:raise AssertionError('Matched pair type differs')
    expected={(m,c,'structural_'+k) for m in LABELS for c in ('full','last32') for k in ('long','short')}
    if set(cells)!=expected or any(set(v)!=set(classes) for v in cells.values()):raise AssertionError('Complete paired grid required')
    groups={t:sorted(g for g,c in classes.items() if c==t) for t in TYPES}
    if any(not x for x in groups.values()):raise AssertionError('All three bracket types required')
    rng=np.random.default_rng(seed);indices={t:rng.integers(len(gs),size=(replicates,len(gs))) for t,gs in groups.items()}
    def values(model,condition,kind,metric=0):
        cell=cells[(model,condition,'structural_'+kind)]
        return {t:np.array([cell[g][metric] for g in gs]) for t,gs in groups.items()}
    def subtract(a,b):return {t:a[t]-b[t] for t in TYPES}
    def estimate(v):
        draws=sum(v[t][indices[t]].mean(1) for t in TYPES)/3
        return dict(mean=float(sum(v[t].mean() for t in TYPES)/3),conditional_95_interval=np.quantile(draws,[.025,.975]).tolist())
    means={m:{c:{k:dict(nll=estimate(values(m,c,k)),accuracy=estimate(values(m,c,k,1))) for k in ('long','short')} for c in ('full','last32')} for m in LABELS}
    contrasts={}
    for model in ('ordinary','pause'):
        long=subtract(values(model,'full','long'),values('latent','full','long'))
        short=subtract(values(model,'full','short'),values('latent','full','short'))
        contrasts['long_vs_'+model]=estimate(long);contrasts['short_vs_'+model]=estimate(short)
        contrasts['long_minus_short_vs_'+model]=estimate(subtract(long,short))
        contrasts['extra_context_utility_vs_'+model]=estimate(subtract(
            subtract(values('latent','last32','long'),values('latent','full','long')),
            subtract(values(model,'last32','long'),values(model,'full','long'))))
    for model in LABELS:
        for kind in ('long','short'):contrasts['full_context_utility_'+model+'_'+kind]=estimate(subtract(values(model,'last32',kind),values(model,'full',kind)))
    return dict(macro_type_weighting='one third per closer type; source-group bootstrap stratified by type',
        groups_by_type={t:len(g) for t,g in groups.items()},means=means,contrasts=contrasts,
        useful_long_type_gates={m:contrasts['long_vs_'+m]['mean']>=.03 and contrasts['long_vs_'+m]['conditional_95_interval'][0]>0 for m in ('ordinary','pause')},
        uniform_type_prior_nll=math.log(3),uniform_type_prior_accuracy=1/3)


def guard(root,p):
    if time.time()>p['absolute_deadline_unix'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3:raise RuntimeError('CPU time/RSS bound')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<512*1024**3 or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Protected host/output reserves')


def checkpoint(root,base,arm):
    path=root/base/'plan.json';plan=json.loads(path.read_text());a=plan['arms'][arm];update=a['checkpoint_updates'][-1]
    f=root/'runs'/a['phase']/f'PRIVATE-u{update}.pt';audit=root/base/('audit-'+arm)/'report.json';r=json.loads(audit.read_text());h=file_sha256(f)
    if r['status']!='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS' or {v['checkpoint_sha256'] for v in r['rows'] if v['update']==update}!={h}:
        raise AssertionError('Audited original endpoint required')
    return dict(plan_relative=str(path.relative_to(root)),plan_sha256=file_sha256(path),arm=arm,
        checkpoint_relative=str(f.relative_to(root)),checkpoint_sha256=h,audit_sha256=file_sha256(audit))


def prepare(root):
    import numpy as np
    from tokenizers import Tokenizer
    if (root/TIED/'analysis/report.json').exists():raise AssertionError('Freeze before tied challenger quality scores')
    panel=root/'probability-structural-panel-v1';pr=json.loads((panel/'report.json').read_text())
    if pr['status']!='FROZEN_STRUCTURAL_DEPENDENCY_PANEL' or pr['paired_source_groups']!=141:raise AssertionError('Original complete source panel')
    if any(file_sha256(panel/n)!=h for n,h in pr['files'].items()):raise AssertionError('Original panel bytes changed')
    mp=root/'interleaved-mixed-round-tuning-v1/plan.json';modelplan=json.loads(mp.read_text())
    exclusions=json.loads((root/modelplan['data_relative']/'plan.json').read_text())['excluded_group_files']
    if exclusions['probability-structural-panel-v1/PRIVATE-metadata.json']!=pr['files']['PRIVATE-metadata.json']:raise AssertionError('Exact TRAIN source exclusions')
    sp=root/'interleaved-mixed-sequential-dev-v1/plan.json';selection=json.loads(sp.read_text())['selected_arm']
    if json.loads((sp.parent/'report.json').read_text())['status']!='COMPLETE_SEQUENTIAL_DEV_READOUT':raise AssertionError('Completed sequential candidate required')
    points={'latent':checkpoint(root,'interleaved-mixed-round-tuning-v1',selection),'pause':checkpoint(root,'interleaved-lr-tuning-v1','pause-lr1')}
    tokenizer=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json');t=Tokenizer.from_file(str(tokenizer))
    classes={c:[] for c in TYPES}
    for i in range(t.get_vocab_size()):
        s=t.decode([i],skip_special_tokens=False).lstrip()
        if s and s[0] in classes:classes[s[0]].append(i)
    meta=json.loads((panel/'PRIVATE-metadata.json').read_text());y=np.load(panel/'targets.npy');mask=np.load(panel/'mask.npy')
    for i,item in enumerate(meta):
        positions=np.flatnonzero(mask[i])
        if len(positions)!=1 or int(y[i,positions[0]]) not in classes[item['closer_type']]:raise AssertionError('Every authentic target must map to its original bracket type')
    files=[str(f) for f in Path('compute_trainium').glob('*.py')]
    files+=['scripts/readout_interleaved_bracket_type.py','scripts/readout_interleaved_structural.py','scripts/evaluate_structural_probability.py',
            'docs/INTERLEAVED_BRACKET_TYPE_20261029.md','tests/test_interleaved_bracket_type.py','tests/test_interleaved_last_readout.py']
    now=time.time();p=dict(identity=ID,registered_unix=now,absolute_deadline_unix=now+3600,points=points,
        pending_tied_plan_sha256=file_sha256(root/TIED/'plan.json'),pending_control='Previously registered broad-NLL-selected tied ordinary endpoint; no structural selection.',
        panel_relative=panel.name,panel_report_sha256=file_sha256(panel/'report.json'),panel_files=pr['files'],
        tokenizer_sha256=file_sha256(tokenizer),class_counts={c:len(v) for c,v in classes.items()},
        source_sha256={f:file_sha256(f) for f in files},bootstrap_seed=10033,bootstrap_replicates=2000,
        optimizer_updates=0,assigned_chips=0,protected_groups_scored=0,
        scope='Post-lead exploratory natural-code type diagnostic, frozen before stronger-control quality. One seed,reused source panel, equal data. Conditional type NLL and accuracy are not general reasoning or an efficiency result; original broad comparison remains primary.')
    guard(root,p);out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'PRIVATE-classes.json',classes)
    p['classes_sha256']=file_sha256(out/'PRIVATE-classes.json');write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def run(root):
    import numpy as np
    import torch
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.tied_interleaved_ordinary import make_model as tied_model
    from compute_trainium.interleaved_last_readout import cached_last_logits
    from compute_trainium.interleaved_sequential_reference import cached_sequential
    from scripts.evaluate_structural_probability import truncated_arrays
    from scripts.readout_interleaved_structural import statistics
    out=root/ID;p=json.loads((out/'plan.json').read_text());start=time.time();rows=[];checks=[]
    result=dict(status='RUNNING_BRACKET_TYPE_DIAGNOSTIC',assigned_chips=0,optimizer_updates=0,protected_groups_scored=0)
    try:
        if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Frozen diagnostic source changed')
        panel=root/p['panel_relative']
        if file_sha256(panel/'report.json')!=p['panel_report_sha256'] or any(file_sha256(panel/n)!=h for n,h in p['panel_files'].items()):raise AssertionError('Frozen panel changed')
        if file_sha256(out/'PRIVATE-classes.json')!=p['classes_sha256']:raise AssertionError('Tokenizer class map changed')
        if file_sha256(root/TIED/'plan.json')!=p['pending_tied_plan_sha256']:raise AssertionError('Control scientific plan changed')
        classes=json.loads((out/'PRIVATE-classes.json').read_text());class_ids={c:torch.tensor(classes[c]) for c in TYPES}
        torch.set_num_threads(8);torch.set_num_interop_threads(1)
        xs=np.load(panel/'inputs.npy');ys=np.load(panel/'targets.npy');ms=np.load(panel/'mask.npy');meta=json.loads((panel/'PRIVATE-metadata.json').read_text())
        panels={'full':(xs,ys,ms),'last32':truncated_arrays(xs,ys,ms,meta)}
        with torch.inference_mode():
            for label in LABELS:
                guard(root,p)
                if label=='ordinary':
                    report=Path('results/interleaved_workspace_20261029/tied-ordinary-tuning-v1/report.json')
                    while not report.exists():guard(root,p);time.sleep(5)
                    if json.loads(report.read_text())['status']!='COMPLETE_TIED_ORDINARY_TUNING':raise AssertionError('Control queue stopped; no substitute endpoint')
                    selected=json.loads(report.with_name('analysis.json').read_text())['selection']
                    point=checkpoint(root,TIED,selected['arm']);write_state(out/'control-admission.json',dict(selection=selected,point=point))
                else:point=p['points'][label]
                if file_sha256(root/point['plan_relative'])!=point['plan_sha256'] or file_sha256(root/point['checkpoint_relative'])!=point['checkpoint_sha256']:raise AssertionError('Audited checkpoint/plan changed')
                plan=json.loads((root/point['plan_relative']).read_text());model=(tied_model(plan) if label=='ordinary' else make_model(plan)).double().eval()
                saved=torch.load(root/point['checkpoint_relative'],map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(saved['model']);del saved
                if label=='latent':
                    for i in (0,1):
                        x=torch.from_numpy(np.array(panels['last32'][0][i:i+1,:32],dtype=np.int64))
                        a=cached_last_logits(model,x);b=cached_sequential(model,x)[:,-1]
                        error=float((a.log_softmax(-1)-b.log_softmax(-1)).abs().max());checks.append(dict(index=i,maximum_log_probability_error=error))
                        if error>1e-8:raise AssertionError('Final-only readout parity')
                for condition,(xx,yy,mm) in panels.items():
                    for i,item in enumerate(meta):
                        guard(root,p);pos=np.flatnonzero(mm[i])
                        if len(pos)!=1:raise AssertionError('One original target required')
                        j=int(pos[0]);x=torch.from_numpy(np.array(xx[i:i+1,:j+1],dtype=np.int64));target=int(yy[i,j])
                        logits=cached_last_logits(model,x) if label=='latent' else model(x,0,label)[:,-1]
                        z=logits[0];log_masses=torch.stack([torch.logsumexp(z[class_ids[c]],0) for c in TYPES]);gold=TYPES.index(item['closer_type'])
                        raw=float(torch.logsumexp(z,0)-z[target]);typed=float(torch.logsumexp(log_masses,0)-log_masses[gold])
                        rows.append(dict(label=label,condition=condition,id=item['id'],group=item['group'],kind=item['kind'],closer=item['closer_type'],
                            targets=1,loss_sum=raw,type_nll=typed,type_correct=int(int(log_masses.argmax())==gold)))
                        if i%16==0:write_state(out/'progress.json',dict(label=label,condition=condition,completed_cases=i+1,wall_seconds=time.time()-start))
                    write_state(out/'PRIVATE-scores.json',rows)
                del model
        result.update(status='COMPLETE_BRACKET_TYPE_DIAGNOSTIC',raw=statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed']),
            typed=typed_statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed']),checks=checks,trained_seeds=1,
            selected_control=selected,scope=p['scope'],automatic_promotion=False)
    except BaseException as error:
        result.update(status='STOPPED_BRACKET_TYPE_DIAGNOSTIC',error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a.root)
