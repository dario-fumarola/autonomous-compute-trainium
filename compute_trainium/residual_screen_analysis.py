"""Fixed-mixture paired source-group analysis of a frozen-predictor screen."""
import json
import numpy as np


def contrast(rows,values):
 groups=sorted({r['group'] for r in rows});lookup={g:i for i,g in enumerate(groups)};num=np.zeros((len(groups),3));den=np.zeros_like(num)
 for r,x in zip(rows,values,strict=True):i=lookup[r['group']];num[i,r['domain']]+=x;den[i,r['domain']]+=1
 weights=np.array([.6,.2,.2]);rng=np.random.default_rng(7286);boot=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=5000);d=boot@den;valid=(d>0).all(1);samples=((boot[valid]@num)/d[valid])@weights
 return dict(mean=float((num.sum(0)/den.sum(0))@weights),conditional_source_group_95_interval=np.quantile(samples,[.025,.975]).tolist(),groups=len(groups),valid_draws=int(valid.sum()),scope='Reused development sources,one branch seed,frozen backbone;not fresh confirmation or seed uncertainty.')


def analyze(out,plan):
 data={arm:json.loads((out/f'PRIVATE-{arm}-scores.json').read_text()) for arm in plan['arm_modes']};rows=data['two'];scores={};comparisons={}
 for arm,values in data.items():
  if [r['index'] for r in values]!=list(range(448)):raise AssertionError('Complete frozen evaluation required')
  if max(abs(a['raw_nll']-b['raw_nll']) for a,b in zip(rows,values,strict=True))>1e-7:raise AssertionError('A branch changed the raw predictor')
  scores[arm]={k:contrast(values,[r[k] for r in values])['mean'] for k in ['raw_nll','workspace_nll']};comparisons[arm+'_utility']=contrast(values,[r['raw_nll']-r['workspace_nll'] for r in values])
  if abs(scores[arm]['raw_nll']-2.282111260797275)>1e-5:raise AssertionError('Original C0 raw32 must reproduce')
 for arm in ['linear','one']:comparisons[arm+'_minus_two']=contrast(rows,[a['workspace_nll']-b['workspace_nll'] for a,b in zip(data[arm],rows,strict=True)])
 u=comparisons['two_utility'];promising=u['mean']>=.01 and u['conditional_source_group_95_interval'][0]>0 and all(comparisons[a+'_minus_two']['mean']>=.005 and comparisons[a+'_minus_two']['conditional_source_group_95_interval'][0]>0 for a in ['linear','one'])
 return dict(scores=scores,contrasts=comparisons,promising_iterative_mechanism=promising,efficiency_claim=False,from_scratch=False,scope='Equal source/anchor exposure,not matched total training compute.Only branch parameters trained;all foundation/feature/head costs remain real.No tuning,early stopping,adaptive allocation or unsupported depth readout.')
