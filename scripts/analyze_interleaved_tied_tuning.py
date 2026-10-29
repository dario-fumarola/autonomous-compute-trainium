"""Select the tied conventional challenger on its own frozen LR grid."""
import argparse,json,time
from pathlib import Path
from scripts.analyze_probability_screen import score


def main(a):
    start=time.time();p=json.loads(a.plan.read_text());folder=a.root/'runs'/(p['identity']+'-evaluation')
    if json.loads((folder/'report.json').read_text())['status']!='COMPLETE_INTERLEAVED_SCREEN_EVALUATION':
        raise AssertionError('Complete independent readout gate required')
    rows=[r for rank in range(64) for r in json.loads((folder/f'PRIVATE-rank{rank}-scores.json').read_text())]
    scores={}
    for name,arm in p['arms'].items():
        chosen=[r for r in rows if r['label']==name+'-main' and r['panel']=='broad']
        if len(chosen)!=384 or len({r['id'] for r in chosen})!=384:raise AssertionError('Exact frozen panel')
        domains={d:score([r for r in chosen if r['domain']==d]) for d in ('web','math','code')}
        scores[name]=dict(lr=arm['lr'],domains=domains,broad_nll=sum(w*domains[d] for d,w in zip(('web','math','code'),(.6,.2,.2))))
    candidates=sorted(scores,key=lambda n:scores[n]['lr'])
    if [scores[n]['lr'] for n in candidates]!=p['learning_rate_grid']:raise AssertionError('Frozen LR opportunity')
    best=min(candidates,key=lambda n:(scores[n]['broad_nll'],scores[n]['lr']))
    result=dict(status='COMPLETE_TIED_ORDINARY_TUNING_ANALYSIS',scores=scores,
        selection=dict(arm=best,**scores[best],bracketed=0<candidates.index(best)<len(candidates)-1),
        assigned_chips=0,protected_groups_scored=0,trained_seeds=1,automatic_promotion=False,wall_seconds=time.time()-start,
        scope='Single-seed conventional-challenger LR selection on reused DEV. No unbiased confirmation or full conventional frontier claim. Primary latent comparison requires its exact sequential readout, not its better finite-round approximation.')
    a.output.mkdir(exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
