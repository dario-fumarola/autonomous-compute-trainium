"""All declared untied seed readouts and the single-pair resource control."""
import argparse
import json
from pathlib import Path
from compute_trainium.confirmation import endpoint_inventory,paired_seed_interval
from compute_trainium.metrics import DocumentScore,paired_group_interval


def comparisons(rows,scores):
    by_id={r['identity']:r for r in rows}
    gains=lambda x,y:{str(s):by_id[f'{x}-1b-s{s}']['metrics']['weighted_nll']-
        by_id[f'{y}-1b-s{s}']['metrics']['weighted_nll'] for s in [23,47,71]}
    result={}
    for x,y in [('ordinary','shared'),('ordinary','untied'),('shared','untied')]:
        values=gains(x,y)
        result[f'{x}_to_{y}_three_seeds']=paired_seed_interval(values)
        if y=='untied':
            result[f'{x}_to_{y}_fresh_seeds']=paired_seed_interval({s:values[s] for s in ['47','71']})
    pairs={}
    for name,x,y in [('ordinary_to_shared_3b','ordinary-3b-s23','shared-3b-s23'),
        ('ordinary_matched_flops_to_untied_1b','ordinary-flop-s23','untied-1b-s23')]:
        pairs[name]=paired_group_interval(scores[x],scores[y],draws=2000,seed=23)
    return {'paired_seed_comparisons':result,'single_pair_diagnostics':pairs,
        'scope':'The ordinary/shared original three-seed result is primary. Seed23 motivated untied replication; combined untied intervals are descriptive. Fresh seeds47/71 have df1. Source-group intervals condition on fitted weights; they do not estimate training-seed variance.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    follow=json.loads((a.root/'confirmation-campaign-v3/followups/report.json').read_text())
    if follow['status']!='COMPLETE_CONFIRMATION_FOLLOWUPS':raise ValueError('Follow-up endpoints incomplete')
    rows=[];scores={}
    for model in endpoint_inventory('/work/instance-nvme',follow['specification']['flop_match']['updates']):
        phase=a.root.parent/Path(model['checkpoint']).relative_to('/work')
        train=json.loads(phase.with_name('report.json').read_text());dev=Path(str(phase.parent)+'-dev')
        report=json.loads((dev/'report.json').read_text())
        rows.append({**{k:model[k] for k in ['identity','arm','seed','endpoint']},'metrics':report['metrics'],
            'training_real_targets':train['total_real_targets'],'parameters':train['parameters'],
            'training_matmul_flops':train['total_training_matmul_flops']})
        scores[model['identity']]=[DocumentScore(**s) for s in json.loads((dev/'document-scores.json').read_text())]
    stats=comparisons(rows,scores)
    result={'status':'COMPLETE_UNTIED_REPLICATION_AND_COMPUTE_CONTROL','rows':rows,**stats,
        'flop_match':follow['specification']['flop_match'],'measured_budget':follow['specification']['budget']}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    text='''# Untied replication and annealed compute control

Two fresh seeds use the previously selected recipe, without further tuning.
Seed 23 is the discovery model and is retained. Every endpoint completes its
declared decay. The untied model has 251.1M parameters; ordinary/shared have
150.4M. No synthetic research examples, teacher or RL enter this comparison.

| Model | Seed | Endpoint | Real targets | DEV NLL |
| --- | ---: | --- | ---: | ---: |
'''
    for r in rows:text+=f"| {r['arm']} | {r['seed']} | {r['endpoint']} | {r['training_real_targets']:,} | {r['metrics']['weighted_nll']:.6f} |\n"
    text+='\nPositive gain favors the second model.\n\n| Paired comparison | Mean gain | 95% training-seed interval |\n| --- | ---: | ---: |\n'
    for key,v in stats['paired_seed_comparisons'].items():
        text+=f"| {key} | {v['mean_gain_nats']:.6f} | [{v['interval_95'][0]:.6f}, {v['interval_95'][1]:.6f}] |\n"
    text+='\n| Single pair | Gain | 95% source-group interval conditional on weights |\n| --- | ---: | ---: |\n'
    for key,v in stats['single_pair_diagnostics'].items():text+=f"| {key} | {v['gain_nats']:.6f} | [{v['interval_95'][0]:.6f}, {v['interval_95'][1]:.6f}] |\n"
    text+='''
The original ordinary/shared three-seed outcome remains primary. The combined
untied interval includes its discovery seed; the separate fresh-seed interval
has only two fits and one degree of freedom. A token-matched gain for the larger
untied network is not a parameter- or compute-matched win. The ordinary resource
control matches estimated cumulative dense training matmul work, including the
vocabulary projection, with integer-update rounding. It does not equalize Adam,
communication, compilation or measured chip-hours. The resource comparison is
one seed; its group bootstrap cannot establish robustness to training seeds.
All domain scores and exact work counts are retained in the machine-readable
readout. TEST is a separate frozen confirmation, never a selection panel.

[Protocol](../../docs/CONFIRMATION_FOLLOWUP_20261023.md) · [Full readout](summary.json).
'''
    (a.output/'REPORT.md').write_text(text);print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
