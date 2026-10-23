"""Single-pair sharing and longer-exposure readouts, preserving the 1B primary."""
import argparse,json
from pathlib import Path
from compute_trainium.metrics import DocumentScore,paired_group_interval


def scores(root,run):
 return [DocumentScore(**r) for r in json.loads((root/'runs'/(run+'-dev')/'document-scores.json').read_text())]


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 old=a.root.parent;rows=[];contrasts={};untied=None;longer=None
 up=a.root/'untied-depth-v1/report.json';lp=a.root/'stable-continuation-v2/report.json'
 if up.exists():
  untied=json.loads(up.read_text())
  if untied['status']=='COMPLETE_UNTIED_SINGLE_SEED_CONTROL':
   final=next(r for r in untied['records'] if r['update']==3840)
   for label,root,name in [('ordinary',old,'natural-curve-v1-ordinary-s23-u3840'),('shared',a.root,'preln-recurrent-v1-s23-u3840')]:
    baseline=json.loads((root/'runs'/(name+'-dev')/'report.json').read_text())
    contrasts[label+'_to_untied_1b']=paired_group_interval(scores(root,name),scores(a.root,'untied-depth-v1-s23-u3840'),draws=2000,seed=23)
   for label,root,name in [('ordinary',old,'natural-curve-v1-ordinary-s23-u3840'),('shared',a.root,'preln-recurrent-v1-s23-u3840'),('untied',a.root,'untied-depth-v1-s23-u3840')]:
    train=json.loads((root/'runs'/name/'report.json').read_text());dev=json.loads((root/'runs'/(name+'-dev')/'report.json').read_text())
    rows.append({'arm':label,'seed':23,'endpoint':'1B','real_targets':train['total_real_targets'],'parameters':train['parameters'],
      'nll':dev['metrics']['weighted_nll'],'domains':dev['metrics']['domains'],'training_matmul_flops':train['total_training_matmul_flops']})
 if lp.exists():
  longer=json.loads(lp.read_text())
  if longer['status']=='COMPLETE_SINGLE_PAIR_3B_CONTINUATION':
   for r in longer['records']:
    rows.append({'arm':r['arm'],'seed':23,'endpoint':'3B' if r['update']==11520 else 'stable',
      'update':r['update'],'real_targets':r['real_tokens'],'nll':r['metrics']['weighted_nll'],'domains':r['metrics']['domains'],
      'parameters':150436608,'training_matmul_flops':r['training_matmul_flops']})
   contrasts['ordinary_to_shared_3b']=paired_group_interval(scores(a.root,'stable-cont-v2-ordinary-s23-u11520'),scores(a.root,'stable-cont-v2-recurrent-s23-u11520'),draws=2000,seed=23)
 result={'status':'COMPLETE_SHARING_AND_EXPOSURE_READOUT' if untied and longer and len(rows)>=9 else 'PARTIAL_READOUT',
  'rows':rows,'contrasts':contrasts,'untied_selected_lr':json.loads((a.root/'untied-depth-v1/selected-lr.json').read_text()) if (a.root/'untied-depth-v1/selected-lr.json').exists() else None,
  'scope':'One paired seed23 control and continuation; source-group intervals condition on fitted models, not seed variance. Untied has more parameters, equal shared-path dense matmul FLOPs and matched initial function. Original three-seed 1B outcome remains unchanged.'}
 for folder,key in [('untied-specialization-v1','component_projection'),('core-difference-rank-v1','difference_spectra')]:
  path=a.root/folder/'report.json'
  if path.exists():result[key]=json.loads(path.read_text())
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 text='''# Sharing and longer-exposure controls

These are single-seed diagnostics following the completed three-seed negative
result. The untied core removes parameter sharing and adds capacity while
preserving the initial two-pass function. Its 251,124,480 unique parameters
compare with150,436,608 for ordinary/shared paths. It has the same estimated
dense matmul FLOPs as shared recurrence at equal source exposure; optimizer
and communication costs differ. Each architecture has its own equal-budget
three-candidate LR selection. No teacher, synthetic research corpus or RL.

| Arm | Endpoint | Real targets | Parameters | NLL |
| --- | --- | ---: | ---: | ---: |
'''
 for r in rows:
  text+=f"| {r['arm']} | {r['endpoint']} | {r['real_targets']:,} | {r['parameters']:,} | {r['nll']:.6f} |\n"
 text+='\nPositive gain below means the candidate has lower NLL than its reference.\n\n| Reference to candidate | Gain | 95% source-group interval, conditional on weights |\n| --- | ---: | ---: |\n'
 for label,v in contrasts.items():text+=f"| {label} | {v['gain_nats']:.6f} | [{v['interval_95'][0]:.6f}, {v['interval_95'][1]:.6f}] |\n"
 text+='''
All comparisons use identical held-out source targets and the fixed60/20/20
web/math/code aggregate. The shared/untied1B contrast tests the sharing plus
capacity constraint; it cannot isolate gradient conflict. The3B comparison
resumes both untouched stable900M states, with every Adam moment, RNG, cursor
and past-FLOP counter preserved. Original64-rank plain-AdamW execution and
selected LRs remain unchanged. The append-only source prefix is validated
byte-for-byte. Additional native qualification updates are discarded before
restarting from the original parents and count as research cost.

Stable checkpoints are descriptive. The1B and3B endpoints each complete their
own final decay. These are different horizon branches; a larger exposure does
not establish convergence. A single pair has no estimate of seed variation.
Source-group bootstrap intervals cannot replace independent training seeds.
The original three-seed1B primary and its negative decision remain unchanged.
All training and readout costs, including tuning and discarded qualification,
are retained separately from reused controls. TEST confirmation is a separate
frozen release after model/analysis identities are fixed.

[Machine-readable readout](summary.json) · [Frozen campaign](../../docs/MECHANISM_DIAGNOSIS_20261023.md).
'''
 for x,y in [('with150','with 150'),('fixed60','fixed 60'),('shared/untied1B','shared/untied 1B'),('The3B','The 3B'),('stable900M','stable 900M'),('Original64','Original 64'),('The1B','The 1B'),('and3B','and 3B'),('three-seed1B','three-seed 1B')]:text=text.replace(x,y)
 if 'component_projection' in result:
  text+='\nA separate read-only diagnosis uses twelve previously fixed DEV prefixes, with512 real targets each. It averages the selected family of corresponding weights between the two trained core copies, without reoptimization. This is a perturbation of one fitted model, not a retrained sharing control or a TEST candidate.\n\n| Projection | Fixed-prefix NLL |\n| --- | ---: |\n'
  for row in result['component_projection']['rows']:text+=f"| {row['mode']} | {row['weighted_nll']:.6f} |\n"
  text+='\nAveraging can change parameter norms and disrupt coadapted features. These effects do not add linearly or establish which constraint caused the training deficit. Separate attention and MLP values matter strongly to this fitted network; this does not rule out learning a better shared solution from scratch. No projected weights are saved or evaluated on TEST.\n'
 if 'difference_spectra' in result:
  import statistics
  spectra=result['difference_spectra']['rows']
  energy=statistics.mean(row['energy_fraction_by_rank']['16'] for row in spectra)
  rank=statistics.median(row['rank_for_80_percent_energy'] for row in spectra)
  text+=f'\nExact SVD on every attention/MLP matrix in four preselected, evenly spaced core layers finds rank16 captures {energy:.1%} of copy-difference energy on average; the median rank for80% energy is {rank:g}. This is weight-space geometry, not a functional compression guarantee or evidence that low-rank adapters cannot learn another solution.\n'
 text=text.replace('with512','with 512').replace('rank16','rank 16').replace('for80%','for 80%')
 (a.output/'REPORT.md').write_text(text)
 print(json.dumps({'status':result['status'],'rows':len(rows),'contrasts':contrasts},indent=2))


if __name__=='__main__':main()
