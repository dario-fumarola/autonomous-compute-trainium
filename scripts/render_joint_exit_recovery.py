"""Render complete FP64 recovery without erasing the failed Native readout."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path('results/probability_workspace_20261029/joint-exit-readout-recovery-v1')
r=json.loads((D/'report.json').read_text())
if r['status']!='COMPLETE_FP64_JOINT_EXIT_RECOVERY':raise AssertionError('Complete recovered analysis required')
labels=[('untied-ordinary-work-k0','Ordinary, matched matrix work'),('untied-ordinary-data-k0','Ordinary, equal data'),('untied-one-k1','Final-only workspace'),('joint-k0','Joint, shallow exit'),('joint-k1','Joint, refined exit'),('joint-permuted-k1','Joint, permuted values')]
lines=['# Joint-exit objective: complete FP64 readout','',
 'One new 79.18M model, 200,190,896 natural targets, complete cooldown and passing mature training audits. All six endpoints are recomputed in FP64 using the same source targets. Lower NLL is better.','',
 '| Predictor | Broad NLL | Web | Math | Code |','|---|---:|---:|---:|---:|']
for key,name in labels:
    s=r['scores'][key];lines.append('| '+name+' | '+' | '.join(f'{x:.6f}' for x in [s['broad_nll'],*[s['domains'][d] for d in ('web','math','code')]])+' |')
lines+=['','Positive differences favor the joint/refined model as named. Intervals are conditional on these weights and the reused DEV source groups; they do not measure training-seed uncertainty.','',
 '| Comparison | Gain | Conditional 95% interval |','|---|---:|---:|']
for key,c in r['comparisons'].items():
    v=c['broad'];lo,hi=v['conditional_95_interval'];lines.append(f"| {key} | {v['gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
lines+=['',f"Registered development gates: `{r['screening_gates']}`.",'',
 'Every real target has total weight one: half on each supported exit. Both exits receive ordinary NTP, and gradients remain connected through the generated proposal. Initialization, parameters, source order, LR and endpoint match the historical final-only candidate. Matrix work is unchanged.','',
 'The original Native evaluation stopped on the permuted-value condition: maximum target error 0.000222091 versus the 0.000200 limit. Its partial scores are not used. No tolerance was relaxed and no checkpoint was changed. The CPU recovery recomputes the candidate and every reused control in FP64 within the original campaign deadline.','',
 'This single-seed, early-learning comparison uses one common LR and a repeatedly inspected panel. A better shallow-to-deep difference cannot replace an absolute ordinary-model advantage. Permutation damage alone is not reasoning evidence. All authored-answer and domain results remain in the complete analysis. Both fresh source reserves remain unscored.','',
 'The new fit used 4.573590 assigned chip-hours; the failed Native readout used 0.727966. CPU recovery has zero assigned accelerator time. Qualification, historical controls and instance-opportunity costs remain in their separate ledgers and must not be double-counted.','',
 '[Complete analysis](report.json) · [Recovery costs](execution-costs.json) · [Frozen recovery](plan.json) · [Original stopped history](../joint-exit-screen-v1/report.json)','']
(D/'REPORT.md').write_text('\n'.join(lines))
keys=['ordinary_equal_work','ordinary_equal_data','final_only_objective','supported_refinement']
names=['vs ordinary, matched work','vs ordinary, equal data','vs final-only workspace','refined vs shallow, same model']
fig,ax=plt.subplots(figsize=(9,3.8))
for i,key in enumerate(keys):
    v=r['comparisons'][key]['broad'];m=v['gain'];lo,hi=v['conditional_95_interval'];ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color='#2166ac')
ax.axvline(0,color='#333',lw=1);ax.axvline(.01,color='#888',ls='--',lw=1)
ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set_xlabel('NLL advantage (positive favors refined joint-exit prediction)')
ax.set_title('Joint-exit screen • all endpoints FP64 • one development seed');ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(D/'comparison.png',dpi=180);fig.savefig(D/'comparison.svg');plt.close(fig)
