"""Token-weighted paired source-group inference on frozen dependency cohorts."""
import json
import numpy as np


def interval(rows,values,*,draws=5000,seed=5721):
    groups=sorted({r['group'] for r in rows});lookup={g:i for i,g in enumerate(groups)};num=np.zeros(len(groups));den=np.zeros(len(groups))
    for row,value in zip(rows,values,strict=True):
        w=row['real_targets'];num[lookup[row['group']]]+=value*w;den[lookup[row['group']]]+=w
    if not len(groups) or den.sum()<=0:raise ValueError('Real source targets required')
    rng=np.random.default_rng(seed);boot=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=draws);samples=(boot@num)/(boot@den)
    return dict(mean=float(num.sum()/den.sum()),conditional_source_group_95_interval=np.quantile(samples,[.025,.975]).tolist(),source_groups=len(groups),cases=len(rows),real_targets=int(den.sum()),draws=draws,warning='Conditional on fixed trained models and reused development sources;not training-seed or calibration uncertainty.No confirmation alpha spent.')


def analyze(root,plan):
    from pathlib import Path
    folder=Path(root)/'runs'/(plan['identity']+'-evaluation');cases=json.loads((Path(root)/plan['panel_relative']).read_text());data={};orders=[('original','full'),('original','short'),('strict','full'),('strict_rms','full')]
    for arm in ['c0','ordinary','latent','constant']:
        rows=[r for k in range(64) for r in json.loads((folder/f'rank{k}-{arm}-metrics.json').read_text())];data[arm]={}
        for variant,condition in orders:
            values=sorted([x for x in rows if x['variant']==variant and x['condition']==condition],key=lambda r:r['index'])
            if [r['index'] for r in values]!=list(range(len(cases))):raise AssertionError('Exact panel and intervention coverage required')
            if any(r['id']!=cases[i]['id'] or r['group']!=cases[i]['group'] or int(r['real_targets'])!=len(cases[i]['answer']) for i,r in enumerate(values)):raise AssertionError('Original task identity or real-target count changed')
            data[arm][variant+':'+condition]=values
        for variant in ['strict','strict_rms']:
            if max(abs(a['raw_nll']-b['raw_nll']) for a,b in zip(data[arm]['original:full'],data[arm][variant+':full'],strict=True))>1e-5:raise AssertionError('Mechanism control changed the ordinary prediction')
    result={};decisions={}
    for kind in sorted({c['kind'] for c in cases}):
        result[kind]={}
        for cohort in [0,1]:
            indices=[i for i,c in enumerate(cases) if c['kind']==kind and c['cohort']==cohort];rows=[data['latent']['original:full'][i] for i in indices];cohort_result=dict(coverage=dict(cases=len(rows),source_groups=len({r['group'] for r in rows}),adequate=len(rows)>=32 and len({r['group'] for r in rows})>=12),models={},contrasts={})
            def vals(arm,view,key):return [data[arm][view][i][key] for i in indices]
            def contrast(name,left,right):cohort_result['contrasts'][name]=interval(rows,[a-b for a,b in zip(left,right,strict=True)])
            contrast('ordinary_context_gain',vals('ordinary','original:short','raw_nll'),vals('ordinary','original:full','raw_nll'))
            contrast('c0_context_gain',vals('c0','original:short','raw_nll'),vals('c0','original:full','raw_nll'))
            for variant in ['original','strict','strict_rms']:
                view=variant+':full';contrast(variant+'_latent_utility',vals('latent',view,'raw_nll'),vals('latent',view,'workspace_nll'));contrast(variant+'_ordinary_minus_latent',vals('ordinary','original:full','raw_nll'),vals('latent',view,'workspace_nll'));contrast(variant+'_constant_minus_latent',vals('constant',view,'workspace_nll'),vals('latent',view,'workspace_nll'))
                if variant!='original':contrast('original_minus_'+variant,vals('latent','original:full','workspace_nll'),vals('latent',view,'workspace_nll'))
            for arm in data:
                cohort_result['models'][arm]={}
                for view in data[arm]:
                    selected=[data[arm][view][i] for i in indices];total=sum(r['real_targets'] for r in selected);score={}
                    for field in selected[0]:
                        if field in ['index','id','kind','cohort','group','condition','variant','real_targets']:continue
                        score[field]=(sum(r[field] for r in selected)/len(selected) if field in ['raw_exact','workspace_exact','raw_first_nll','workspace_first_nll'] else sum(r[field]*r['real_targets'] for r in selected)/total)
                    cohort_result['models'][arm][view]=score
            result[kind][str(cohort)]=cohort_result
        if kind in ['code_binding','math_conclusion']:
            variants={}
            for variant in ['original','strict','strict_rms']:
                checks=[]
                for c in result[kind].values():
                    context=c['contrasts']['ordinary_context_gain'];work=c['contrasts'][variant+'_latent_utility'];checks.append(c['coverage']['adequate'] and context['mean']>=.1 and context['conditional_source_group_95_interval'][0]>0 and work['mean']>=.03 and work['conditional_source_group_95_interval'][0]>0)
                variants[variant]=dict(replicated_point_and_interval_gate=all(checks),primary=(variant=='original'),scope='Strict and RMS controls are exploratory interventions,not rescue of original quality/compute claim.')
            decisions[kind]=variants
    return dict(status='COMPLETE_FROZEN_DEPENDENCY_ANALYSIS',families=result,decisions=decisions,confirmation=False,optimizer_updates=0,terminal_alpha_spent=0,scope='Primary original-reader results and all pre-panel mechanism controls shown separately.Two source-group cohorts are development replication,not independent training seeds.Positive context use is required before interpreting lack of workspace benefit.')
