"""Render final, audited scores without further source reads."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results/probability_workspace_20261029/untied-screen-v1')
r=json.loads((p/'report.json').read_text())
if r['status']!='COMPLETE_UNTIED_PROBABILITY_SCREEN':raise AssertionError('Completed audited screen required')
a=json.loads((p/'analysis.json').read_text());s=a['scores'];cs=a['comparisons']
fig,ax=plt.subplots(figsize=(9,4.8));labels=['Ordinary, equal data','Fixed one workspace','Ordinary, equal work'];y=list(range(3))
for values,color,name,shift in [(['ordinary_data-k0','fixed-final-k1','ordinary_work-k0'],'#a16207','Tied tables (historical)',-.1),(['untied-ordinary-data-k0','untied-one-k1','untied-ordinary-work-k0'],'#2563eb','Independent tables (new)',.1)]:
 ax.scatter([s[k]['broad_nll'] for k in values],[v+shift for v in y],label=name,color=color,s=55)
 for k,yy in zip(values,y):ax.annotate(f"{s[k]['broad_nll']:.5f}",(s[k]['broad_nll'],yy+shift),xytext=(5,-2),textcoords='offset points',fontsize=9,color=color)
ax.set(yticks=y,yticklabels=labels,xlabel='Development NLL (lower is better)',title='Probability workspace and its ordinary controls');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(loc='best');ax.margins(x=.25,y=.2)
fig.tight_layout();fig.savefig(p/'comparison.png',dpi=160);fig.savefig(p/'comparison.svg');plt.close(fig)
lines=['# Independent prediction/value tables: controlled learning screen','',
 '| Model | Tied tables,62.8M | Independent tables,79.18M |','| --- | ---: | ---: |']
for title,left,right in zip(labels,['ordinary_data-k0','fixed-final-k1','ordinary_work-k0'],['untied-ordinary-data-k0','untied-one-k1','untied-ordinary-work-k0']):lines.append(f"| {title} | {s[left]['broad_nll']:.6f} | {s[right]['broad_nll']:.6f} |")
lines+=['','Positive gains below favor the workspace or the untying intervention.','',
 '| Contrast | Gain | Conditional95% source-group interval |','| --- | ---: | --- |']
for k,title in [('ordinary_equal_data','Workspace vs new equal-data ordinary'),('ordinary_equal_work','Workspace vs new equal-work ordinary'),('workspace_untying','Workspace improvement from untying'),('ordinary_data_untying','Ordinary equal-data improvement from untying'),('untying_interaction_equal_data','Change in workspace-minus-ordinary advantage, equal data'),('untying_interaction_equal_work','Change in workspace-minus-ordinary advantage, equal work')]:
 d=cs[k]['broad'];lo,hi=d['conditional_95_interval'];lines.append(f"| {title} | {d['gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
lines+=['','![Comparison](comparison.png)','',
 'Three new fits share the same79,184,384parameters,initialization,source order and',
 'uniform NTP objective. Fixed-one and ordinary equal-data see200,190,896 real',
 'targets;ordinary equal-work sees446,570,790. All endpoints complete cooldown.',
 'Only the one-replay exit of the workspace model is trained and supported.',
 'Old tied scores are reused byte-identically. They are historical controls,',
 'not new seed replication. Untying adds16,384,000parameters to both new models.',
 'The cross-generation interaction includes that intervention;it is not a',
 'fixed-total-parameter capacity estimate. A relative recovery is insufficient',
 'without the absolute ordinary-control comparisons.', '',
 'All independent mature gradients,full-state/replica checks and FP64 readout',
 'gates precede quality scoring. One common preregistered LR is a screening',
 'recipe,not the independently tuned conventional frontier. These intervals',
 'condition on one trained seed and a repeatedly inspected development panel.',
 'All domains and authentic dependency targets remain in the full analysis.',
 'The320 protected confirmation groups remain sealed.', '',
 'There is no supported within-checkpoint iteration curve or learned allocation.',
 'Semantic permutation is an OOD intervention,not reasoning evidence by itself.',
 'Matrix work includes vocabulary projections but excludes elementwise,optimizer',
 'and communication work;assigned chip-time is reported separately. No cached',
 'decoding-speed claim is made. No automatic larger fit or confirmation follows.', '',
 '[Frozen protocol](../../../docs/UNTIED_PROBABILITY_SCREEN_20261029.md) ·',
 '[All scores](analysis.json) · [Audits](report.json) · [Execution costs](execution-costs.json).']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
