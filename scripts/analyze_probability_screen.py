"""All frozen comparisons, with source-group conditional uncertainty."""
import argparse,collections,json,time
from pathlib import Path
import numpy as np


def score(rows,field='loss_sum'):
    if not rows:return None
    return sum(r[field] for r in rows)/(len(rows) if field=='first_loss' else sum(r['targets'] for r in rows))


def paired(left,right,*,seed,replicates,weighted=False,field='loss_sum'):
    """Positive means right has lower loss. Reuse group draws within each pair."""
    l={r['id']:r for r in left};r={x['id']:x for x in right}
    if l.keys()!=r.keys():raise AssertionError('Comparison examples differ')
    domains=['web','math','code'] if weighted else ['all'];weights=[.6,.2,.2] if weighted else [1.]
    rng=np.random.default_rng(seed);draws=np.zeros(replicates);mean=0.
    for domain,weight in zip(domains,weights):
        groups=collections.defaultdict(lambda:np.zeros(3))
        for key,a in l.items():
            b=r[key]
            if (a['group'],a['targets'])!=(b['group'],b['targets']):raise AssertionError('Paired source targets differ')
            if weighted and a['domain']!=domain:continue
            groups[a['group']]+=np.array([a[field],b[field],1 if field=='first_loss' else a['targets']])
        values=np.array(list(groups.values()))
        if not len(values):return None
        mean+=weight*(values[:,0].sum()-values[:,1].sum())/values[:,2].sum()
        chosen=rng.integers(len(values),size=(replicates,len(values)));sums=values[chosen].sum(1)
        draws+=weight*(sums[:,0]-sums[:,1])/sums[:,2]
    return dict(gain=float(mean),conditional_95_interval=np.quantile(draws,[.025,.975]).tolist(),cases=len(l),groups=len({x['group'] for x in left}))


def main(a):
    start=time.time();p=json.loads(a.plan.read_text());folder=a.root/'runs'/(p['identity']+'-evaluation')
    if json.loads((folder/'report.json').read_text())['status']!='COMPLETE_PROBABILITY_SCREEN_EVALUATION':raise AssertionError('Audited evaluation required')
    rows=[x for rank in range(64) for x in json.loads((folder/f'PRIVATE-rank{rank}-scores.json').read_text())]
    labels=sorted({r['label'] for r in rows});panels={'broad':[r for r in rows if r['panel']=='broad']}
    for kind in ('code_binding','math_conclusion','math_intermediate_exploratory'):
        panels[kind]=[r for r in rows if r['kind']==kind]
    scores={}
    for label in labels:
        selected=[r for r in panels['broad'] if r['label']==label];domains={d:score([r for r in selected if r['domain']==d]) for d in ('web','math','code')}
        scores[label]=dict(broad_nll=sum(w*domains[d] for d,w in zip(('web','math','code'),(.6,.2,.2))),domains=domains,
            dependency={kind:dict(answer_nll=score([r for r in rs if r['label']==label]),first_target_nll=score([r for r in rs if r['label']==label],'first_loss'),cases=sum(r['label']==label for r in rs)) for kind,rs in panels.items() if kind!='broad'})
    pairs={
      'final_extra_1':('workspace-final-k0','workspace-final-k1'),
      'final_extra_2':('workspace-final-k1','workspace-final-k2'),
      'final_extra_total':('workspace-final-k0','workspace-final-k2'),
      'early_extra_1':('workspace-early-k0','workspace-early-k1'),
      'early_extra_2':('workspace-early-k1','workspace-early-k2'),
      'ordinary_equal_data':('ordinary-data-final-k0','workspace-final-k2'),
      'ordinary_equal_work':('ordinary-work-final-k0','workspace-final-k2'),
      'constant_proposal_control':('uniform-final-k2','workspace-final-k2'),
      'proposal_content_intervention':('workspace-final-permuted-k2','workspace-final-k2'),
      'ordinary_early_equal_data':('ordinary-data-early-k0','workspace-early-k2'),
      'no_replay_vs_ordinary':('ordinary-data-final-k0','workspace-final-k0')}
    comparisons={}
    for name,(left,right) in pairs.items():
        comparisons[name]={}
        for panel,rs in panels.items():
            ls=[r for r in rs if r['label']==left];rr=[r for r in rs if r['label']==right]
            kw=dict(seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates'],weighted=panel=='broad')
            comparisons[name][panel]=paired(ls,rr,**kw)
            if panel!='broad':comparisons[name][panel+'_first_target']=paired(ls,rr,field='first_loss',**kw)
        comparisons[name]['domains']={d:paired([r for r in panels['broad'] if r['label']==left and r['domain']==d],[r for r in panels['broad'] if r['label']==right and r['domain']==d],seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates']) for d in ('web','math','code')}
    passed=lambda name:comparisons[name]['broad']['gain']>=p['practical_gain_nats'] and comparisons[name]['broad']['conditional_95_interval'][0]>0
    result=dict(status='COMPLETE_PROBABILITY_SCREEN_ANALYSIS',scores=scores,comparisons=comparisons,
        screening_gates=dict(useful_supported_second_step=passed('final_extra_2'),useful_vs_equal_data=passed('ordinary_equal_data'),useful_vs_equal_work=passed('ordinary_equal_work')),
        trained_seeds=1,protected_groups_scored=0,automatic_promotion=False,assigned_chips=0,wall_seconds=time.time()-start,
        interpretation='All intervals condition on trained weights and source-group sampling; not training-seed uncertainty or confirmation. Early checkpoints are uncooled descriptive points. All final endpoints cooled independently. Constant-proposal control is cheaper. Semantic permutation is an out-of-distribution intervention,not proof of reasoning. Nine code-name cases are not a reasoning benchmark.')
    a.output.mkdir(exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
