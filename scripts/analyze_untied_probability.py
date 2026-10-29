"""Untied-versus-tied contrasts; never replace the new ordinary controls."""
import argparse,json,time
from pathlib import Path
from scripts.analyze_probability_screen import score,paired
from compute_trainium.training import file_sha256
PAIRS={
 'ordinary_equal_data':('untied-ordinary-data-k0','untied-one-k1'),
 'ordinary_equal_work':('untied-ordinary-work-k0','untied-one-k1'),
 'workspace_untying':('fixed-final-k1','untied-one-k1'),
 'ordinary_data_untying':('ordinary_data-k0','untied-ordinary-data-k0'),
 'ordinary_work_untying':('ordinary_work-k0','untied-ordinary-work-k0'),
 'proposal_content':('untied-one-permuted-k1','untied-one-k1')}

def interaction_rows(rows,ordinary):
    """(Untied ordinary - untied workspace) - (tied ordinary - tied workspace)."""
    labels=[ordinary,'fixed-final-k1','untied-one-k1','ordinary_data-k0' if ordinary=='untied-ordinary-data-k0' else 'ordinary_work-k0']
    sets=[{r['id']:r for r in rows if r['label']==n} for n in labels]
    if not all(x.keys()==sets[0].keys() for x in sets):raise AssertionError('Interaction examples must match')
    left=[];right=[]
    for key in sets[0]:
        rs=[s[key] for s in sets]
        if len({(r['group'],r['targets'],r['domain']) for r in rs})!=1:raise AssertionError('Interaction grouping/targets must match')
        l=dict(rs[0]);r=dict(rs[2])
        for field in ('loss_sum','first_loss'):
            l[field]=rs[0][field]+rs[1][field];r[field]=rs[2][field]+rs[3][field]
        left.append(l);right.append(r)
    return left,right

def main(a):
    start=time.time();p=json.loads(a.plan.read_text());folder=a.root/'runs'/(p['identity']+'-evaluation')
    if json.loads((folder/'report.json').read_text())['status']!='COMPLETE_PROBABILITY_SCREEN_EVALUATION':raise AssertionError('Audited new readout required')
    parent=a.root/'runs'/p['parent_evaluation_identity']
    if any(file_sha256(parent/n)!=h for n,h in p['parent_score_sha256'].items()):raise AssertionError('Historical score bytes changed')
    load=lambda f:[r for rank in range(64) for r in json.loads((f/f'PRIVATE-rank{rank}-scores.json').read_text())]
    new,old=load(folder),load(parent)
    if {r['label'] for r in new}&{r['label'] for r in old}:raise AssertionError('Distinct model labels required')
    rows=new+old;panels={'broad':[r for r in rows if r['panel']=='broad']}
    panels.update({k:[r for r in rows if r['kind']==k] for k in ('code_binding','math_conclusion','math_intermediate_exploratory')})
    scores={}
    for label in sorted({r['label'] for r in rows}):
        ds={d:score([r for r in panels['broad'] if r['label']==label and r['domain']==d]) for d in ('web','math','code')}
        scores[label]=dict(broad_nll=sum(w*ds[d] for d,w in zip(('web','math','code'),(.6,.2,.2))),domains=ds,
          dependency={k:dict(answer_nll=score([r for r in rs if r['label']==label]),first_target_nll=score([r for r in rs if r['label']==label],'first_loss'),cases=sum(r['label']==label for r in rs)) for k,rs in panels.items() if k!='broad'})
    comparisons={}
    names=list(PAIRS)+['untying_interaction_equal_data','untying_interaction_equal_work']
    def selection(name,rs):
        if name in PAIRS:
            l,r=PAIRS[name];return [x for x in rs if x['label']==l],[x for x in rs if x['label']==r]
        return interaction_rows(rs,'untied-ordinary-data-k0' if name.endswith('data') else 'untied-ordinary-work-k0')
    for name in names:
        cs={}
        for panel,rs in panels.items():
            l,r=selection(name,rs);kw=dict(seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates'],weighted=panel=='broad')
            cs[panel]=paired(l,r,**kw)
            if panel!='broad':cs[panel+'_first_target']=paired(l,r,field='first_loss',**kw)
        cs['domains']={d:paired(*selection(name,[r for r in panels['broad'] if r['domain']==d]),seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates']) for d in ('web','math','code')}
        comparisons[name]=cs
    passes=lambda n:comparisons[n]['broad']['gain']>=p['practical_gain_nats'] and comparisons[n]['broad']['conditional_95_interval'][0]>0
    result=dict(status='COMPLETE_UNTIED_PROBABILITY_ANALYSIS',scores=scores,comparisons=comparisons,
      screening_gates={n:passes(n) for n in ('ordinary_equal_data','ordinary_equal_work')},
      trained_seeds=1,new_fits=3,protected_groups_scored=0,reused_scores_sha256_verified=True,automatic_promotion=False,assigned_chips=0,wall_seconds=time.time()-start,
      interpretation='Fixed-one exit only supported. All new models have79,184,384parameters and identical initialization. Unchanged200M workspace/data exposure and matched matrix-work ordinary control;every endpoint cooled. Historical tied scores reused byte-identically;interactions are conditional source comparisons,not independent replication or fixed-parameter effects of untying. One common LR;not the tuned conventional frontier. No iteration,allocation or thinking claim from one exit or OOD corruption. All320 protected source groups remain sealed.')
    a.output.mkdir(exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],gates=result['screening_gates'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
