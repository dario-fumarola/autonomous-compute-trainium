"""Render all predeclared joint-exit outcomes after audited analysis."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('results/probability_workspace_20261029/joint-exit-screen-v1')
r=json.loads((D/'analysis.json').read_text());cost=json.loads((D/'execution-costs.json').read_text());p=json.loads((D/'plan.json').read_text())
labels=[('untied-ordinary-work-k0','Ordinary, matched matrix work'),('untied-ordinary-data-k0','Ordinary, equal data'),('untied-one-k1','Final-only workspace'),('joint-k0','Joint, shallow exit'),('joint-k1','Joint, refined exit'),('joint-permuted-k1','Joint, permuted values')]
text=['# Joint-exit objective: completed early-learning screen','',
 'One new79.18M model,200,190,896 natural targets and complete cooldown. All64 replicas,state restoration,mature gradients and FP64 readouts pass. Historical controls are byte-pinned,not new seeds. Lower NLL is better.','',
 '| Predictor | Broad NLL | Web | Math | Code |','|---|---:|---:|---:|---:|']
for key,name in labels:
 s=r['scores'][key];text.append('| '+name+' | '+' | '.join(f'{x:.6f}' for x in [s['broad_nll'],*[s['domains'][d] for d in ('web','math','code')]])+' |')
text+=['','Positive differences below favor the refined/joint model as named. Intervals condition on these weights and reused DEV source groups,not training-seed variation.','',
 '| Comparison | Gain | Conditional95% interval |','|---|---:|---:|']
for name,c in r['comparisons'].items():
 v=c['broad'];a,b=v['conditional_95_interval'];text.append(f"| {name} | {v['gain']:+.6f} | [{a:+.6f}, {b:+.6f}] |")
text+=['',f"Registered development gates: `{r['screening_gates']}`.",'',
 'Both0/1 exits receive uniform natural-token supervision,with half-weight per exit. Total weight at each real target remains one. The generated proposal remains fully differentiable. Dispatched matrix work equals the final-only recipe;runtime costs are separately recorded.','',
 'A gain over the same checkpoint’s shallow exit cannot replace an absolute ordinary-model win. Permutation damage is an out-of-distribution intervention and cannot prove reasoning. All domain and authored-answer contrasts remain in analysis.json;no favorable subset replaces the broad endpoint.','',
 'This is one seed,one common LR,an early exposure and a repeatedly used DEV panel. A conventional intermediate-supervision control,fresh source groups and independent seeds remain necessary for stronger attribution or confirmation. No autonomous allocation or two-step claim. All320 protected groups remain sealed.','',
 '[Complete analysis](analysis.json) · [Audits](report.json) · [Costs](execution-costs.json) · [Frozen plan](plan.json)','']
(D/'REPORT.md').write_text('\n'.join(text))
keys=['ordinary_equal_work','ordinary_equal_data','final_only_objective','supported_refinement']
names=['vs ordinary, matched work','vs ordinary, equal data','vs final-only workspace','refined vs shallow, same model']
fig,ax=plt.subplots(figsize=(9,3.7))
for i,k in enumerate(keys):
 v=r['comparisons'][k]['broad'];mean=v['gain'];lo,hi=v['conditional_95_interval'];ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,color='#2166ac')
ax.axvline(0,color='#333',lw=1);ax.axvline(.01,color='#777',ls='--',lw=1)
ax.set_yticks(range(len(keys)),names);ax.invert_yaxis();ax.set_xlabel('NLL advantage (positive favors refined joint-exit prediction)')
ax.set_title('Joint-exit screen • one seed • conditional DEV intervals');ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(D/'comparison.png',dpi=180);fig.savefig(D/'comparison.svg');plt.close(fig)
