"""Select within each equally budgeted LR grid; no confirmation claim."""
import argparse,json,time
from pathlib import Path
from scripts.analyze_probability_screen import score


def main(a):
    started=time.time();p=json.loads(a.plan.read_text());folder=a.root/'runs'/(p['identity']+'-evaluation')
    if json.loads((folder/'report.json').read_text())['status']!='COMPLETE_INTERLEAVED_SCREEN_EVALUATION':raise AssertionError('All mature/readout gates required')
    rows=[r for rank in range(64) for r in json.loads((folder/f'PRIVATE-rank{rank}-scores.json').read_text())]
    results={}
    for name,arm in p['arms'].items():
        selected=[r for r in rows if r['label']==name+'-main' and r['panel']=='broad']
        if len(selected)!=384 or len({r['id'] for r in selected})!=384:raise AssertionError('Exact frozen development panel required')
        domains={d:score([r for r in selected if r['domain']==d]) for d in ('web','math','code')}
        results[name]=dict(family=arm['family'],lr=arm['lr'],domains=domains,broad_nll=sum(w*domains[d] for d,w in zip(('web','math','code'),(.6,.2,.2))))
    choices={}
    for family in ('ordinary','latent','pause'):
        candidates=sorted([dict(arm=n,**r) for n,r in results.items() if r['family']==family],key=lambda r:r['lr'])
        if [r['lr'] for r in candidates]!=p['learning_rate_grid']:raise AssertionError('Equal frozen tuning opportunity')
        best=min(candidates,key=lambda r:(r['broad_nll'],r['lr']));index=candidates.index(best)
        choices[family]=dict(**best,bracketed=0<index<len(candidates)-1,required_next_action=('separate measured main registration' if 0<index<len(candidates)-1 else 'one separately registered outward LR candidate; no automatic promotion'))
    result=dict(status='COMPLETE_INTERLEAVED_LR_TUNING',scores=results,selection=choices,trained_seeds=1,protected_groups_scored=0,automatic_promotion=False,assigned_chips=0,wall_seconds=time.time()-started,
        interpretation='This reused development panel selects learning rates. Scores are not independent confirmation, an unbiased comparison after selection, a thinking result, or a conventional-frontier efficiency claim. All endpoint cooldowns and audits are required.')
    a.output.mkdir(exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
