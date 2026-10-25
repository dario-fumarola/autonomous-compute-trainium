"""Position-preserving full/short source views and paired mechanism metrics."""
import math
import random


def views(tokens,row):
    lo,hi=row['target_start_token'],row['target_end_token']
    full,short=row['full_start_token'],row['short_start_token']
    if not 0<=full<=short==lo-128<lo<hi<=len(tokens):raise ValueError('Invalid original source target/context span')
    result={}
    for name,start in [('full',full),('short',short)]:
        inputs=tokens[start:hi-1];positions=list(range(start-full,hi-1-full))
        predictions=list(range(lo-start-1,hi-start-1))
        if not inputs or len(inputs)>1024 or len(predictions)!=hi-lo:raise ValueError('Context/target geometry exceeds frozen window')
        result[name]={'inputs':inputs,'targets':tokens[lo:hi],'positions':positions,
                      'prediction_positions':predictions,'state_start':start-full}
    if result['full']['inputs'][short-full:]!=result['short']['inputs']:
        raise AssertionError('Short input changed retained source tokens')
    return result


def summarize(rows):
    bands=['129-256','257-512','513-900'];totals={}
    for band in bands:
        selected=[r for r in rows if r['band']==band]
        n=sum(r['real_targets'] for r in selected)
        if not n:raise ValueError('All frozen distance bands require targets')
        losses={key:sum(r['nll_sum'][key] for r in selected)/n for key in ['full1','full2','short1','short2']}
        gain=losses['full1']-losses['full2'];short=losses['short1']-losses['short2']
        totals[band]={'nll':losses,'full_gain':gain,'short_gain':short,'interaction':gain-short,
                      'real_targets':n,'identifiers':len(selected)}
    return {'bands':totals,'full_gain':sum(r['full_gain'] for r in totals.values())/3,
        'interaction':sum(r['interaction'] for r in totals.values())/3,
        'aggregation':'Equal one-third distance bands; real identifier-token NLL within each band.'}


def group_intervals(rows,draws=5000,seed=1381):
    groups=sorted({r['group'] for r in rows});rng=random.Random(seed);values=[]
    if len(groups)<5:raise ValueError('Too few repository groups')
    grouped={g:[r for r in rows if r['group']==g] for g in groups}
    for _ in range(draws):
        sample=[r for g in rng.choices(groups,k=len(groups)) for r in grouped[g]]
        try:score=summarize(sample)
        except ValueError:continue
        values.append((score['full_gain'],score['interaction']))
    if len(values)<.9*draws:raise ValueError('Sparse repository bootstrap across distance bands')
    intervals={}
    for i,key in enumerate(['full_gain','interaction']):
        ordered=sorted(r[i] for r in values)
        intervals[key]=[ordered[int(.025*len(ordered))],ordered[min(len(ordered)-1,int(.975*len(ordered)))]]
    return {'intervals_95':intervals,'repository_groups':len(groups),'accepted_draws':len(values),
        'warning':'Nominal within-fitted-model repository uncertainty only. Not independent training-seed significance,terminal alpha or a new confirmation panel.'}
