"""Source-volume versus repeated-update controls,with the original cheap comparators."""
import json
from .residual_screen_analysis import contrast


def analyze(out,plan):
 data={arm:json.loads((out/f'PRIVATE-{arm}-scores.json').read_text()) for arm in plan['arm_modes']};rows=data['fresh_two'];scores={};comparisons={}
 for arm,values in data.items():
  if [r['index'] for r in values]!=list(range(448)) or any(r['group']!=plan['evaluation_groups'][i] for i,r in enumerate(values)):raise AssertionError('Exact reused evaluation panel required')
  if max(abs(a['raw_nll']-b['raw_nll']) for a,b in zip(rows,values,strict=True))>1e-7:raise AssertionError('An arm changed the base predictor')
  scores[arm]={k:contrast(values,[r[k] for r in values])['mean'] for k in ['raw_nll','workspace_nll']};comparisons[arm+'_utility']=contrast(values,[r['raw_nll']-r['workspace_nll'] for r in values])
  if abs(scores[arm]['raw_nll']-2.282111260797275)>1e-5:raise AssertionError('Original frozen predictor must reproduce')
 def difference(name,left,right):comparisons[name]=contrast(rows,[a['workspace_nll']-b['workspace_nll'] for a,b in zip(data[left],data[right],strict=True)])
 for mode in ['linear','one','two']:difference('fresh_minus_repeat_gain_'+mode,'repeat_'+mode,'fresh_'+mode)
 for schedule in ['fresh','repeat']:
  for mode in ['linear','one']:difference(schedule+'_'+mode+'_minus_two',schedule+'_'+mode,schedule+'_two')
 u=comparisons['fresh_two_utility'];mech=u['mean']>=.01 and u['conditional_source_group_95_interval'][0]>0 and all(comparisons['fresh_'+m+'_minus_two']['mean']>=.005 and comparisons['fresh_'+m+'_minus_two']['conditional_source_group_95_interval'][0]>0 for m in ['linear','one'])
 volume=comparisons['fresh_minus_repeat_gain_two'];vg=volume['mean']>=.005 and volume['conditional_source_group_95_interval'][0]>0
 return dict(scores=scores,contrasts=comparisons,promising_iterative_mechanism=mech,fresh_source_gate=vg,efficiency_claim=False,from_scratch=False,scope='Same trained prefix and exact2048total updates,fresh8192views versus2048views repeated four times.Same sampled positions,RNG and optimizer schedule.All previous512updates count.One seed/reused development panel;no confirmation.Source gain is separate from the iteration gain.')
