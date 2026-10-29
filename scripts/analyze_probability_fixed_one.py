"""Fixed-one follow-up: all registered comparisons and repeated-readout parity."""
import argparse,json,time
from pathlib import Path
from scripts.analyze_probability_screen import score,paired
from compute_trainium.training import file_sha256

PAIRS={
 'ordinary_equal_work':('ordinary_work-k0','fixed-final-k1'),
 'ordinary_equal_data':('ordinary_data-k0','fixed-final-k1'),
 'sampled_same_exit':('sampled-final-k1','fixed-final-k1'),
 'sampled_deeper_exit':('sampled-final-k2','fixed-final-k1'),
 'constant_two_additions':('uniform-k2','fixed-final-k1'),
 'proposal_content':('fixed-final-permuted-k1','fixed-final-k1')}
REUSED={'sampled-final-k1':'workspace-final-k1','sampled-final-k2':'workspace-final-k2',
 'ordinary_data-k0':'ordinary-data-final-k0','ordinary_work-k0':'ordinary-work-final-k0','uniform-k2':'uniform-final-k2'}

def repeat_error(rows,previous):
    old={(x['label'],x['panel'],x['id']):x for x in previous};errors=[]
    selected=[x for x in rows if x['label'] in REUSED]
    expected={(REUSED[x['label']],x['panel'],x['id']) for x in selected}
    available={key for key in old if key[0] in REUSED.values()}
    if expected!=available:raise AssertionError('Reused readouts cover different sources')
    for x in selected:
        y=old[(REUSED[x['label']],x['panel'],x['id'])]
        if (x['group'],x['targets'],x['kind'],x['domain'])!=(y['group'],y['targets'],y['kind'],y['domain']):raise AssertionError('Reused target definition changed')
        errors.extend([abs(x['loss_sum']-y['loss_sum'])/x['targets'],abs(x['first_loss']-y['first_loss'])])
    if not errors:raise AssertionError('Missing reused controls')
    return max(errors)

def main(a):
    start=time.time();p=json.loads(a.plan.read_text());folder=a.root/'runs'/(p['identity']+'-evaluation')
    if json.loads((folder/'report.json').read_text())['status']!='COMPLETE_PROBABILITY_SCREEN_EVALUATION':raise AssertionError('Audited evaluation required')
    oldfolder=a.root/'runs'/p['parent_evaluation_identity']
    if any(file_sha256(oldfolder/name)!=h for name,h in p['parent_score_sha256'].items()):raise AssertionError('Parent evaluation changed')
    load=lambda f:[x for rank in range(64) for x in json.loads((f/f'PRIVATE-rank{rank}-scores.json').read_text())]
    rows=load(folder);previous=load(oldfolder);error=repeat_error(rows,previous)
    if error>p['evaluation_maximum_loss_error']:raise AssertionError('Repeated control readout mismatch')
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
    result=dict(status='COMPLETE_FIXED_ONE_ANALYSIS',scores=scores,comparisons=comparisons,
      screening_gates={n:passes(n) for n in ('ordinary_equal_work','ordinary_equal_data','sampled_same_exit')},
      reused_readout_maximum_error=error,trained_seeds=1,new_fits=1,protected_groups_scored=0,automatic_promotion=False,assigned_chips=0,wall_seconds=time.time()-start,
      interpretation='Only fixed-one exit is supported. Reused controls and development sources are not independent replication. Same work/data comparison tests fixed versus sampled depth;it does not isolate one specific optimization cause. Constant control has two mean additions,not one,so it is not an exact matched mechanism control. Source-group intervals condition on these weights. Content corruption is out of distribution,not proof of reasoning.')
    a.output.mkdir(exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],gates=result['screening_gates'],repeated_control_error=error)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
