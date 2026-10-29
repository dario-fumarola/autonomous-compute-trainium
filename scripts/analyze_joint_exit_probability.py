"""Registered supported-exit and absolute comparisons; existing controls retained."""
import argparse,json,math,time
from pathlib import Path
from scripts.analyze_probability_screen import score,paired
from compute_trainium.training import file_sha256
PAIRS={
 'supported_refinement':('joint-k0','joint-k1'),
 'ordinary_equal_data':('untied-ordinary-data-k0','joint-k1'),
 'ordinary_equal_work':('untied-ordinary-work-k0','joint-k1'),
 'final_only_objective':('untied-one-k1','joint-k1'),
 'shallow_vs_ordinary_data':('untied-ordinary-data-k0','joint-k0'),
 'shallow_vs_ordinary_work':('untied-ordinary-work-k0','joint-k0'),
 'proposal_content_intervention':('joint-permuted-k1','joint-k1')}

def analyze(rows,p):
    labels=set(r['label'] for r in rows);seen=set()
    for r in rows:
        key=(r['label'],r['panel'],r['id'])
        if key in seen:raise AssertionError('Duplicate prediction row')
        seen.add(key)
        if r['targets']<=0 or not all(math.isfinite(r[k]) for k in ('loss_sum','first_loss')):raise AssertionError('Finite positive-target scores required')
    required={x for pair in PAIRS.values() for x in pair}
    if not required<=labels:raise AssertionError('Every registered model required')
    panels={'broad':[r for r in rows if r['panel']=='broad']}
    panels.update({k:[r for r in rows if r['kind']==k] for k in ('code_binding','math_conclusion','math_intermediate_exploratory')})
    scores={}
    for label in sorted(labels):
        ds={d:score([r for r in panels['broad'] if r['label']==label and r['domain']==d]) for d in ('web','math','code')}
        scores[label]=dict(broad_nll=sum(w*ds[d] for d,w in zip(('web','math','code'),(.6,.2,.2))),domains=ds,
          dependency={k:dict(answer_nll=score([r for r in rs if r['label']==label]),first_target_nll=score([r for r in rs if r['label']==label],'first_loss'),cases=sum(r['label']==label for r in rs)) for k,rs in panels.items() if k!='broad'})
    cs={}
    for name,(left,right) in PAIRS.items():
        c={}
        for panel,rs in panels.items():
            l=[r for r in rs if r['label']==left];r=[r for r in rs if r['label']==right]
            kw=dict(seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates'],weighted=panel=='broad')
            c[panel]=paired(l,r,**kw)
            if panel!='broad':c[panel+'_first_target']=paired(l,r,field='first_loss',**kw)
        c['domains']={d:paired([r for r in panels['broad'] if r['label']==left and r['domain']==d],[r for r in panels['broad'] if r['label']==right and r['domain']==d],seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates']) for d in ('web','math','code')}
        cs[name]=c
    passes=lambda name:cs[name]['broad']['gain']>=p['practical_gain_nats'] and cs[name]['broad']['conditional_95_interval'][0]>0
    return dict(scores=scores,comparisons=cs,screening_gates={n:passes(n) for n in ('supported_refinement','ordinary_equal_data','ordinary_equal_work')})

def main(a):
    start=time.time();p=json.loads(a.plan.read_text());folder=a.root/'runs'/(p['identity']+'-evaluation')
    if json.loads((folder/'report.json').read_text())['status']!='COMPLETE_PROBABILITY_SCREEN_EVALUATION':raise AssertionError('Audited new readout required')
    parent=a.root/'runs'/p['parent_evaluation_identity']
    if any(file_sha256(parent/n)!=h for n,h in p['parent_score_sha256'].items()):raise AssertionError('Historical score bytes changed')
    load=lambda f:[r for rank in range(64) for r in json.loads((f/f'PRIVATE-rank{rank}-scores.json').read_text())]
    old=[r for r in load(parent) if r['label']!='untied-one-permuted-k1']
    result=dict(status='COMPLETE_JOINT_EXIT_PROBABILITY_ANALYSIS',**analyze(load(folder)+old,p),
      trained_seeds=1,new_fits=1,protected_groups_scored=0,reused_scores_sha256_verified=True,automatic_promotion=False,assigned_chips=0,wall_seconds=time.time()-start,
      interpretation='Both zero and one extra pass supported by uniformly weighted joint NTP. All parameters,initialization,source order,LR and endpoint exposure unchanged versus byte-pinned final-only and ordinary controls. Reused controls are not new replication. Conditional source intervals on reused DEV;one common LR is not a tuned conventional frontier. Same-checkpoint gain alone is insufficient if absolute ordinary comparisons fail. OOD permutation damage is not reasoning evidence. No two-step,allocation or confirmation claim.')
    a.output.mkdir(exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],gates=result['screening_gates'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
