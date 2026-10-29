"""Prespecified K2 LR selection and descriptive comparisons to retained controls."""
import argparse
import json
import time
from pathlib import Path
from scripts.analyze_probability_screen import score


def main(a):
    start=time.time(); p=json.loads(a.plan.read_text())
    folder=a.root/'runs'/(p['identity']+'-evaluation')
    if json.loads((folder/'report.json').read_text())['status']!='COMPLETE_INTERLEAVED_SCREEN_EVALUATION':
        raise AssertionError('Complete numerical readout required')
    rows=[r for rank in range(64) for r in json.loads((folder/f'PRIVATE-rank{rank}-scores.json').read_text())]
    results={}
    for name,arm in p['arms'].items():
        exits={}
        for suffix in ('main','supported3'):
            chosen=[r for r in rows if r['label']==name+'-'+suffix and r['panel']=='broad']
            if len(chosen)!=384 or len({r['id'] for r in chosen})!=384:
                raise AssertionError('Exact frozen broad development panel')
            domains={d:score([r for r in chosen if r['domain']==d]) for d in ('web','math','code')}
            exits[suffix]=dict(domains=domains,broad_nll=sum(w*domains[d] for d,w in zip(('web','math','code'),(.6,.2,.2))))
        results[name]=dict(lr=arm['lr'],exits=exits)
    candidates=sorted(results,key=lambda n:results[n]['lr'])
    if [results[n]['lr'] for n in candidates]!=p['learning_rate_grid']:
        raise AssertionError('Frozen LR opportunity')
    best=min(candidates,key=lambda n:(results[n]['exits']['main']['broad_nll'],results[n]['lr']))
    selected=results[best]; nll=selected['exits']['main']['broad_nll']
    old=json.loads(Path('results/interleaved_workspace_20261029/tuning-v1/analysis.json').read_text())
    gains={family:r['broad_nll']-nll for family,r in old['selection'].items()}
    result=dict(status='COMPLETE_MIXED_ROUND_TUNING_ANALYSIS',scores=results,
        selection=dict(arm=best,lr=selected['lr'],readout='K2 only, prespecified',broad_nll=nll,
            bracketed=0<candidates.index(best)<len(candidates)-1),
        development_gains_over_retained_controls=gains,
        prediction_screen_pass=(gains['ordinary']>=.01 and gains['pause']>=.01),
        sequential_consistency_guard='Required separately for ALL mixed endpoints; no automatic promotion',
        assigned_chips=0,protected_groups_scored=0,trained_seeds=1,wall_seconds=time.time()-start,
        interpretation='Adaptive development follow-up on reused source panel and controls; not independent confirmation, equal-work efficiency or thinking.')
    a.output.mkdir(exist_ok=False); (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
