"""Registered budget-cue comparisons with byte-pinned existing control scores."""
import argparse,json,time
from pathlib import Path
from scripts.analyze_probability_screen import score,paired
from compute_trainium.training import file_sha256

PAIRS={
 'ordinary_equal_work':('ordinary_work-k0','cue-final-k2'),
 'ordinary_equal_data':('ordinary_data-k0','cue-final-k2'),
 'one_vs_ordinary_data':('ordinary_data-k0','cue-final-k1'),
 'one_vs_ordinary_work':('ordinary_work-k0','cue-final-k1'),
 'first_replay':('cue-final-k0','cue-final-k1'),
 'second_replay':('cue-final-k1','cue-final-k2'),
 'matched_fixed_one':('fixed-final-k1','cue-final-k1'),
 'sampled_one':('sampled-final-k1','cue-final-k1'),
 'sampled_two':('sampled-final-k2','cue-final-k2'),
 'constant':('uniform-k2','cue-final-k2'),
 'proposal_content':('cue-final-permuted-k2','cue-final-k2')}

def main(a):
    start=time.time();p=json.loads(a.plan.read_text());folder=a.root/'runs'/(p['identity']+'-evaluation')
    if json.loads((folder/'report.json').read_text())['status']!='COMPLETE_PROBABILITY_SCREEN_EVALUATION':raise AssertionError('Audited evaluation required')
    oldfolder=a.root/'runs'/p['parent_evaluation_identity']
    if any(file_sha256(oldfolder/name)!=h for name,h in p['parent_score_sha256'].items()):raise AssertionError('Parent evaluation changed')
    load=lambda f:[x for rank in range(64) for x in json.loads((f/f'PRIVATE-rank{rank}-scores.json').read_text())]
    rows=load(folder);previous=load(oldfolder)
    if {r['label'] for r in rows}&{r['label'] for r in previous}:raise AssertionError('New and reused labels must be distinct')
    rows+=previous
    panels={'broad':[r for r in rows if r['panel']=='broad']}
    panels.update({kind:[r for r in rows if r['kind']==kind] for kind in ('code_binding','math_conclusion','math_intermediate_exploratory')})
    scores={}
    for label in sorted({r['label'] for r in rows}):
        selected=[r for r in panels['broad'] if r['label']==label];domains={d:score([r for r in selected if r['domain']==d]) for d in ('web','math','code')}
        scores[label]=dict(broad_nll=sum(w*domains[d] for d,w in zip(('web','math','code'),(.6,.2,.2))),domains=domains,
          dependency={kind:dict(answer_nll=score([r for r in rs if r['label']==label]),first_target_nll=score([r for r in rs if r['label']==label],'first_loss'),cases=sum(r['label']==label for r in rs)) for kind,rs in panels.items() if kind!='broad'})
    comparisons={}
    for name,(left,right) in PAIRS.items():
        comparisons[name]={}
        for panel,rs in panels.items():
            ls=[r for r in rs if r['label']==left];rr=[r for r in rs if r['label']==right]
            kw=dict(seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates'],weighted=panel=='broad')
            comparisons[name][panel]=paired(ls,rr,**kw)
            if panel!='broad':comparisons[name][panel+'_first_target']=paired(ls,rr,field='first_loss',**kw)
        comparisons[name]['domains']={d:paired([r for r in panels['broad'] if r['label']==left and r['domain']==d],[r for r in panels['broad'] if r['label']==right and r['domain']==d],seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates']) for d in ('web','math','code')}
    passes=lambda n:comparisons[n]['broad']['gain']>=p['practical_gain_nats'] and comparisons[n]['broad']['conditional_95_interval'][0]>0
    result=dict(status='COMPLETE_BUDGET_CUE_ANALYSIS',scores=scores,comparisons=comparisons,
      screening_gates={n:passes(n) for n in ('ordinary_equal_work','ordinary_equal_data','first_replay','second_replay','matched_fixed_one')},
      reused_scores_sha256_verified=True,trained_seeds=1,new_fits=1,protected_groups_scored=0,automatic_promotion=False,assigned_chips=0,wall_seconds=time.time()-start,
      interpretation='All0/1/2 exits trained. Only remaining-budget cues change versus sampled parent;1024 extra parameters count. Existing control scores are reused byte-for-byte with frozen source/target definitions and previous FP64 audits;not new replication. One seed and repeatedly used development panel. Budget is externally supplied,not learned allocation. Content corruption is out of distribution,not proof of reasoning. No automatic scaling.')
    a.output.mkdir(exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],gates=result['screening_gates'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
