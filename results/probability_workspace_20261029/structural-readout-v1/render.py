import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results/probability_workspace_20261029/structural-readout-v1');r=json.loads((p/'report.json').read_text());assert r['status']=='COMPLETE_STRUCTURAL_PROBABILITY_COMPARISON';a=json.loads((p/'analysis.json').read_text())
lines=['# Source-separated structural dependency readout','',
 'The workspace does not pass either absolute structural-prediction gate. Its',
 'long-target difference from equal-data ordinary is−0.004987nats,with a wide',
 'conditional95%interval[−0.190870,+0.194024]:inconclusive about a small useful',
 'effect. It loses0.340116nats to matched-work ordinary([−0.541180,−0.140772]',
 'when expressed as workspace advantage). No narrow result rescues the broad',
 'negative,and no iteration or autonomous-computation claim is established.','',
 '| Predictor | Long,full prefix | Long,last32 | Nearby,full prefix | Nearby,last32 |',
 '| --- | ---: | ---: | ---: | ---: |']
for key,label in [('workspace','Workspace'),('ordinary_data','Ordinary,equal data'),('ordinary_work','Ordinary,equal work')]:
 v=a['means'][key];lines.append(f"| {label} | {v['full']['long']:.6f} | {v['last32']['long']:.6f} | {v['full']['short']:.6f} | {v['last32']['short']:.6f} |")
lines+=['','Lower NLL is better. All282targets are unchanged authored code tokens,one',
 'long/near pair per141source groups,matched on delimiter type and nesting band.',
 'Groups are disjoint from current TRAIN,earlier evaluated groups and the320',
 'protected confirmation groups. No source program is executed or label generated.',
 'The rules were frozen before the parent comparison’s quality scores.','',
 'The longer prefix helps every model on the long targets. Workspace gains',
 '0.629719nats from full context versus0.392172for equal-data ordinary and',
 '0.627764for equal-work ordinary. The apparently larger workspace context effect',
 'against equal-data ordinary(+0.237548)comes with worse truncated-context',
 'prediction and essentially equal full-context prediction. It is **greater',
 'dependence,not an absolute predictive win**. Removing context changes more',
 'than the matching opener;this does not isolate an internal stack algorithm.',
 'Long/near pairing also does not randomize dependency distance.','',
 'No updates were made. All independent FP64 readout gates pass. The run uses',
 '0.579116assigned chip-hours;data preparation and CPU analysis are separate.',
 'Only the fixed-one workspace exit is trained/supported. These conditional',
 'intervals describe source variation for one seed,not training-run uncertainty.',
 'Source count limits sensitivity;this is a development diagnostic,not a',
 'confirmation result or a universal bound on latent computation.','',
 '[Every contrast](analysis.json) · [Numerical checks](report.json) ·',
 '[Costs](execution-costs.json) · [Rules](prospective-rules.json) ·',
 '[Source reconstruction](../structural-panel-v1/source-reconstruction-audit.json).']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
fig,ax=plt.subplots(figsize=(8,4.5))
for key,label,color in [('workspace','Workspace','#2563eb'),('ordinary_data','Ordinary,equal data','#a16207'),('ordinary_work','Ordinary,equal work','#15803d')]:
 v=a['means'][key];ax.plot([0,1],[v['full']['long'],v['last32']['long']],marker='o',label=label,color=color)
ax.set(xticks=[0,1],xticklabels=['Original prefix (up to512)','Last32tokens'],ylabel='Long-target NLL (lower is better)',title='All models benefit from the longer prefix');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(p/'context-comparison.png',dpi=160);fig.savefig(p/'context-comparison.svg');plt.close(fig)
