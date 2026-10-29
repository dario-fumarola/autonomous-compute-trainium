"""Render completed cue ablation; no extra source reads or model evaluation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results/probability_workspace_20261029/budget-cue-screen-recovery-v1')
r=json.loads((p/'report.json').read_text())
if r['status']!='COMPLETE_BUDGET_CUE_SCREEN':raise AssertionError('Complete audited comparison first')
a=json.loads((p/'analysis.json').read_text());old=json.loads(Path('results/probability_workspace_20261028/evaluation-recovery-v2/analysis.json').read_text())
s=a['scores'];comparisons=a['comparisons']
fig,axes=plt.subplots(1,2,figsize=(12,4.6));left,right=axes
left.plot([0,1,2],[s[f'cue-final-k{k}']['broad_nll'] for k in (0,1,2)],'o-',label='Budget cue, same checkpoint',color='#2563eb')
left.plot([0,1,2],[old['scores'][f'workspace-final-k{k}']['broad_nll'] for k in (0,1,2)],'o--',label='Original sampled model',color='#a16207')
for label,name,color in [('ordinary_data-k0','Ordinary: equal data','#059669'),('ordinary_work-k0','Ordinary: equal work','#111827'),('fixed-final-k1','Separately trained fixed-one','#7c3aed')]:
 left.axhline(s[label]['broad_nll'],label=name,color=color,ls=':')
left.set(xticks=[0,1,2],xlabel='Extra supported computation steps',ylabel='Development NLL (lower is better)',title='Supported depth and ordinary controls')
left.legend(fontsize=8);left.grid(alpha=.15)
items=[('ordinary_equal_work','K2 vs ordinary equal work'),('ordinary_equal_data','K2 vs ordinary equal data'),('second_replay','K2 vs its own K1'),('first_replay','K1 vs its own K0'),('matched_fixed_one','K1 vs fixed-one control')]
for i,(key,label) in enumerate(items):
 d=comparisons[key]['broad'];g=d['gain'];lo,hi=d['conditional_95_interval']
 right.errorbar(g,i,xerr=[[g-lo],[hi-g]],fmt='o',color='#059669' if g>0 else '#dc2626',capsize=3)
right.axvline(0,color='#777',lw=1);right.axvline(.01,color='#2563eb',ls=':',lw=1)
right.set(yticks=list(range(len(items))),yticklabels=[x[1] for x in items],xlabel='Gain in nats (positive favors cue / deeper exit)',title='Registered contrasts; conditional source intervals')
right.invert_yaxis();right.grid(axis='x',alpha=.15)
fig.suptitle('Remaining-budget conditioning: one-seed development ablation',fontsize=12)
fig.tight_layout();fig.savefig(p/'comparison.png',dpi=160);fig.savefig(p/'comparison.svg');plt.close(fig)
lines=['# Remaining-budget cue: completed developmental test','',
 '| Predictor | Weighted development NLL ↓ |','| --- | ---: |']
for label,name in [('cue-final-k0','Cue model, zero extra steps'),('cue-final-k1','Cue model, one extra step'),('cue-final-k2','Cue model, two extra steps'),('fixed-final-k1','Separately trained fixed-one'),('sampled-final-k1','Original sampled model, one extra step'),('ordinary_data-k0','Ordinary, equal data'),('ordinary_work-k0','Ordinary, equal matrix work'),('uniform-k2','Constant proposal')]:
 lines.append(f"| {name} | {s[label]['broad_nll']:.6f} |")
lines+=['','Positive gain below favors the cue model or the deeper supported exit.','',
 '| Contrast | Gain | Conditional95% source-group interval |','| --- | ---: | --- |']
for key,label in items:
 d=comparisons[key]['broad'];lo,hi=d['conditional_95_interval'];lines.append(f"| {label} | {d['gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
lines+=['','![Comparison](comparison.png)','',
 'One new fit from random initialization:1530updates,200,190,896 natural targets,',
 '62,801,408parameters (1024 new cues),unchanged sampled0/1/2 schedule and nominal',
 'matrix-work budget. All three exits are supported. Every endpoint completes',
 'cooldown and independent mature gradients/full-state restoration before scoring.',
 'The old control checkpoints and byte-pinned readouts are reused,not new replication.',
 'The broad384-source-group panel has been repeatedly inspected;intervals condition',
 'on these trained weights and do not include training-seed or model-selection uncertainty.',
 'All320 protected confirmation source groups remain sealed.', '',
 'The matrix ledger excludes elementwise/optimizer work in every arm;cue additions',
 'are not counted as matrices. Runtime and compilation costs remain separate.',
 'A supplied remaining budget is not autonomous allocation. Permuted proposal',
 'values are an OOD intervention,not sufficient evidence of reasoning. The natural',
 'dependency cases and every domain are reported in the complete analysis.', '',
 '[Frozen design](../../../docs/BUDGET_CONDITIONED_SCREEN_20261029.md) ·',
 '[Registration](plan.json) · [All scores and contrasts](analysis.json) ·',
 '[Audits and execution](report.json) · [Costs](execution-costs.json).']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
