"""Publish complete fixed-cell inference and observational state diagnostics."""
import argparse,json,statistics
from pathlib import Path
from compute_trainium.metrics import DocumentScore,paired_group_interval,paired_three_seed_interval


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 report=json.loads((a.root/'unroll-diagnostic-v1/report.json').read_text());rows=report['records']
 cells=[]
 for arm,passes in [('ordinary',1),('ordinary',2),('recurrent',1),('recurrent',2),('recurrent',4)]:
  selected=[r for r in rows if r['arm']==arm and r['inference_passes']==passes]
  if {r['seed'] for r in selected}!={23,47,71}:raise ValueError('Incomplete fixed inference cell')
  cells.append({'training':arm,'inference_passes':passes,'nll':statistics.mean(r['metrics']['weighted_nll'] for r in selected),
   'seed_nll':{str(r['seed']):r['metrics']['weighted_nll'] for r in selected},
   'domain_mean_nll':{d:statistics.mean(r['metrics']['domains'][d]['nll'] for r in selected) for d in ['web','math','code']}})
 contrasts={}
 for name,arm,first,second in [('recurrent_1_to_2','recurrent',1,2),('ordinary_1_to_2','ordinary',1,2),('recurrent_2_to_4','recurrent',2,4)]:
  scores={};gains={}
  for seed in [23,47,71]:
   aa=next(r for r in rows if r['arm']==arm and r['seed']==seed and r['inference_passes']==first)
   bb=next(r for r in rows if r['arm']==arm and r['seed']==seed and r['inference_passes']==second)
   gains[str(seed)]=aa['metrics']['weighted_nll']-bb['metrics']['weighted_nll']
   x=[DocumentScore(**v) for v in json.loads((a.root/'runs'/aa['run']/'document-scores.json').read_text())]
   y=[DocumentScore(**v) for v in json.loads((a.root/'runs'/bb['run']/'document-scores.json').read_text())]
   scores[str(seed)]=paired_group_interval(x,y,draws=2000,seed=23)
  contrasts[name]={**paired_three_seed_interval(gains),'source_group_intervals_by_seed':scores}
 state=json.loads((a.root/'mechanism-states-v1/report.json').read_text());prefix=[]
 for arm in ['ordinary','recurrent']:
  for split in ['train','dev']:
   per_seed={}
   for seed in [23,47,71]:
    items=[r for r in state['records'] if (r['arm'],r['split'],r['seed'])==(arm,split,seed)]
    per_seed[str(seed)]={str(p):sum(w*statistics.mean(r['readouts'][str(p)]['nll'] for r in items if r['domain']==d) for d,w in [('web',.6),('math',.2),('code',.2)]) for p in [1,2]}
   prefix.append({'training':arm,'split':split,'seed_nll':per_seed,'mean_nll':{str(p):statistics.mean(r[str(p)] for r in per_seed.values()) for p in [1,2]}})
 result={'status':'COMPLETE_RECURRENT_MECHANISM_READOUT','cells':cells,'contrasts':contrasts,'fixed_prefix_readout':prefix,
  'gradient_history':json.loads((a.root/'mechanism-diagnostics-v1/gradient-history.json').read_text()),
  'application_gradients':json.loads((a.root/'mechanism-states-v1/application-gradients.json').read_text()),
  'state_probe_specification':state['specification'],'scope':'Exploratory fixed DEV readouts, unchanged weights and positions. Seed and source-group intervals are separate; no inference claim from an untrained path alone.'}
 result['document_boundaries']=json.loads((a.root/'document-boundary-v1/report.json').read_text())
 history_path=a.root/'unroll-development-v1/report.json'
 if history_path.exists():
  history=json.loads(history_path.read_text())
  if history['status']!='COMPLETE_FIXED_PREFIX_UNROLL_DEVELOPMENT':raise ValueError('Incomplete historical unroll readout')
  result['unroll_development']={'rows':history['rows'],'seed':23,'prefixes':12,'real_targets_per_prefix':512,
   'cpu_seconds':history['elapsed_seconds'],
   'scope':'Fixed small DEV prefix panel, not full DEV. Truncated one-pass readout is outside the two-pass training configuration. Shows learned dependence, not a reusable iterative algorithm or an advantage over ordinary.'}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 (a.output/'state-probes.json').write_text(json.dumps(state,indent=2)+'\n')
 text='''# Same-weight recurrence diagnostic

This completes the fixed 15-cell inference matrix on the unchanged curve DEV
panel. Each checkpoint has the same weights and token positions across its
readouts. Distributed trained-path calibration reproduces every original score.
The original three-seed 1B negative result remains unchanged.

| Training path | Inference core passes | Mean NLL |
| --- | ---: | ---: |
'''
 for c in cells:text+=f"| {c['training']} | {c['inference_passes']} | {c['nll']:.6f} |\n"
 text+='\n| Intervention | NLL improvement, first path minus second | 95% paired-seed interval |\n| --- | ---: | ---: |\n'
 for label,key in [('Recurrent checkpoint: 1 to 2','recurrent_1_to_2'),('Ordinary checkpoint: 1 to 2','ordinary_1_to_2'),('Recurrent checkpoint: 2 to 4','recurrent_2_to_4')]:
  v=contrasts[key];text+=f"| {label} | {v['mean_gain_nats']:.6f} | [{v['interval_95'][0]:.6f}, {v['interval_95'][1]:.6f}] |\n"
 text+='''
The recurrent model's one-pass readout is outside its training configuration.
A two-pass advantage establishes dependence on the trained computation path;
it does not establish superior training, an interpretable algorithm, or that
further iterations should improve the prediction. Four passes is a fixed
extrapolation diagnostic. Intervals are exploratory and not simultaneous.

The accompanying fixed-prefix CPU readout uses the same isolated-document
format for TRAIN and DEV, with 512 real targets from each of four source
documents/domain/split. TRAIN prefixes come from the beginning of the actual
frozen training order and are seen by all terminal models. This small sample
is a diagnosis, not a population training/generalization-gap estimate. Compare
the architectures on the same examples; do not interpret a raw TRAIN–DEV level
difference as an unbiased generalization gap.

Global gradient norms were logged before clipping approximately every ten
updates plus phase boundaries. The history therefore estimates sampled
exceedances, not exact per-update clipping frequency. The derivative readout
uses three fixed natural TRAIN prefixes at the unchanged seed23 checkpoint,
with equal-weight copies separating the two applications. Its gradient cosine
is observational and cannot establish that sharing caused the training loss.
State RMS, centered token variation and between-pass changes are reported for
all fixed prefixes. No synthetic research corpus, optimizer update, new LR
selection or TEST scoring is part of these measurements.

[Machine-readable comparisons](summary.json) · [Full state readouts](state-probes.json) ·
[Frozen new campaign](../../docs/MECHANISM_DIAGNOSIS_20261023.md).
'''
 g=result['application_gradients']
 text+=f'\nOn the three inspected TRAIN prefixes, aggregate application-gradient cosine is {g["mean_gradient_cosine"]:.3f}. Residual RMS rises from approximately 1.44 after the first core pass to 2.74 after the second, with substantial centered token variation in both. These summaries provide no evidence of gross token-state collapse; they do not establish a causal optimization explanation.\n'
 boundary=result['document_boundaries']['aggregate'];vals={(r['arm'],r['mode']):r['nll'] for r in boundary}
 text+=f'\nA separate fixed 32-window TRAIN attention intervention retains all tokens, positions and targets. Blocking attention across document boundaries changes ordinary NLL from {vals["ordinary","packed"]:.6f} to {vals["ordinary","document_masked"]:.6f}, and recurrent NLL from {vals["recurrent","packed"]:.6f} to {vals["recurrent","document_masked"]:.6f}. The recurrent deficit remains in both modes. This limited sample does not support boundary context as the sole explanation of the loss. [Boundary protocol](../../docs/DOCUMENT_BOUNDARY_DIAGNOSTIC.md).\n'
 if 'unroll_development' in result:
  history=result['unroll_development']['rows']
  text+='\nA historical read-only check uses the same twelve fixed DEV prefixes (512 real targets each), seed23 only. These are not full-panel scores.\n\n| Training real targets | One-pass NLL | Trained two-pass NLL | Difference |\n| ---: | ---: | ---: | ---: |\n'
  for update in [384,1152,3456,3840]:
   first=next(r for r in history if r['update']==update and r['passes']==1)
   second=next(r for r in history if r['update']==update and r['passes']==2)
   text+=f"| {first['real_targets']:,} | {first['nll']:.6f} | {second['nll']:.6f} | {first['nll']-second['nll']:.6f} |\n"
  text+='\nDependence on the second pass is already present at100M targets and grows on this panel. One pass remains an intervention outside training; this does not establish monotonic refinement, better ordinary prediction or seed robustness. The904M checkpoint is stable; the1B branch is annealed.\n'
  text=text.replace('at100M','at 100M').replace('The904M','The 904M').replace('the1B','the 1B')
 (a.output/'REPORT.md').write_text(text)
 print(json.dumps({'cells':cells,'contrasts':{k:{'mean':v['mean_gain_nats'],'interval':v['interval_95']} for k,v in contrasts.items()}},indent=2))


if __name__=='__main__':main()
