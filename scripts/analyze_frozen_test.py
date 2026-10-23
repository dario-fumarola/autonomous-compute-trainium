"""Frozen analysis: original three-seed confirmation plus declared single pairs."""
import argparse,json
from pathlib import Path
from compute_trainium.metrics import DocumentScore,paired_group_interval,paired_three_seed_interval


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 release=json.loads((a.campaign/'test-release.json').read_text());rows=[];scores={}
 for model in release['models']:
  report=json.loads((a.root/'runs'/model['run']/'report.json').read_text())
  if report['status']!='PASS_SHARDED_TRAINED_PATH_READOUT' or report['split']!='test':raise ValueError('Invalid frozen TEST result')
  rows.append({**{k:model[k] for k in ['identity','arm','seed','endpoint']},'metrics':report['metrics'],'training_real_targets':report['training_real_tokens']})
  scores[model['identity']]=[DocumentScore(**s) for s in json.loads((a.root/'runs'/model['run']/'document-scores.json').read_text())]
 by_id={r['identity']:r for r in rows};gains={str(seed):by_id[f'ordinary-1b-s{seed}']['metrics']['weighted_nll']-by_id[f'shared-1b-s{seed}']['metrics']['weighted_nll'] for seed in [23,47,71]}
 primary=paired_three_seed_interval(gains);lower,upper=primary['interval_95']
 primary['decision']='USEFUL_GAIN_SUPPORTED' if lower>.01 else ('USEFUL_GAIN_RULED_OUT_AT_THIS_EXPOSURE' if upper<.01 else 'INCONCLUSIVE')
 contrasts={}
 for key,x,y in [('shared_to_untied_1b','shared-1b-s23','untied-1b-s23'),('ordinary_to_untied_1b','ordinary-1b-s23','untied-1b-s23'),('ordinary_to_shared_3b','ordinary-3b-s23','shared-3b-s23')]:
  contrasts[key]=paired_group_interval(scores[x],scores[y],draws=2000,seed=23)
 result={'status':'COMPLETE_FROZEN_TEST_CONFIRMATION','rows':rows,'three_seed_1b_confirmation':primary,
  'single_pair_diagnostics':contrasts,'scope':'Original replicated comparison is primary confirmation. Untied and3B have one seed; group intervals condition on weights, not training variance. No model selection or inference-depth intervention on TEST.'}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 text='''# Frozen TEST confirmation

All model identities, trained depths, evaluator and analysis sources were
frozen before opening TEST. This confirms the original three-seed comparison;
the untied control and longer pair remain single-seed diagnostics.

| Model | Seed | Endpoint | TEST weighted NLL |
| --- | ---: | --- | ---: |
'''
 for r in rows:text+=f"| {r['arm']} | {r['seed']} | {r['endpoint']} | {r['metrics']['weighted_nll']:.6f} |\n"
 text+=f"\nOriginal three-seed ordinary-minus-shared gain: **{primary['mean_gain_nats']:.6f} [{lower:.6f}, {upper:.6f}]**. Decision at the declared0.01-nat threshold: `{primary['decision']}`.\n"
 text+='\n| Single-pair comparison | Gain | 95% source-group interval, conditional on weights |\n| --- | ---: | ---: |\n'
 for key,v in contrasts.items():text+=f"| {key} | {v['gain_nats']:.6f} | [{v['interval_95'][0]:.6f}, {v['interval_95'][1]:.6f}] |\n"
 text+='''
Positive gains favor the candidate. Source-group intervals do not replace
independent training seeds. All domain scores are in the machine-readable
report. The original DEV outcome and protocols remain preserved; TEST does
not replace them. No tuning, new model variant, checkpoint selection, or
exploratory unroll uses this panel. Further scientific variants need a
separate held-out panel.

[Machine-readable confirmation](summary.json) · [Frozen release](release.json) ·
[Protocol](../../docs/FROZEN_TEST_CONFIRMATION.md).
'''
 text=text.replace('declared0.01','declared 0.01');(a.output/'REPORT.md').write_text(text)
 (a.output/'release.json').write_text(json.dumps(release,indent=2)+'\n')
 print(json.dumps({'primary':primary,'single_pair_diagnostics':contrasts},indent=2))


if __name__=='__main__':main()
