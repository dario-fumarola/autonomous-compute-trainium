"""Read all predeclared depth cells; never choose a best inference exit."""
import argparse
import json
from pathlib import Path

from compute_trainium.depth_analysis import SEEDS, declared_contrasts
from compute_trainium.confirmation import paired_seed_interval
from compute_trainium.metrics import DocumentScore, paired_group_interval
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    models=json.loads(a.inventory.read_text());rows=[];scores={}
    for model in models:
        for depth in model['allowed_depths']:
            name=model['identity']+f'-{a.split}-r{depth}'
            path=a.root/'runs'/name;report=json.loads((path/'report.json').read_text())
            if report['status']!='PASS_DEPTH_MATRIX_READOUT' or not report['weights_unchanged_all_replicas']:
                raise ValueError('Incomplete/changed-weight depth readout')
            row={k:model[k] for k in ['identity','arm','seed','endpoint','parameters','training_matmul_flops']}
            row.update(depth=depth,metrics=report['metrics'],training_real_targets=report['training_real_tokens'],
                       inference_forward_flops_per_window=report['forward_flops_per_window'])
            rows.append(row);scores[model['identity'],depth]=[DocumentScore(**s) for s in json.loads((path/'document-scores.json').read_text())]
    primary=declared_contrasts(rows);secondary={}
    by_key={(r['arm'],r['seed'],r['endpoint'],r['depth']):r['metrics']['weighted_nll'] for r in rows}
    for name,ref,candidate in [('mixture2_to_mixture3',('mixture',2),('mixture',3)),
          ('ordinary1_to_mixture2_equal_data',('ordinary',1),('mixture',2)),
          ('ordinary1_to_fixed2_equal_data',('ordinary',1),('fixed',2)),
          ('mixture3_to_mixture4_extrapolation',('mixture',3),('mixture',4))]:
        values={str(s):by_key[ref[0],s,'1B',ref[1]]-by_key[candidate[0],s,'1B',candidate[1]] for s in SEEDS}
        secondary[name]=paired_seed_interval(values)
    pairs={}
    longer={m['arm']:m for m in models if m['endpoint']=='3B'}
    if longer:
        for name,ref,candidate in [('ordinary_to_fixed',('ordinary',1),('fixed',2)),
                ('ordinary_to_mixture',('ordinary',1),('mixture',2)),
                ('fixed_to_mixture',('fixed',2),('mixture',2)),
                ('mixture1_to_mixture3',('mixture',1),('mixture',3))]:
            pairs[name]=paired_group_interval(scores[longer[ref[0]]['identity'],ref[1]],
                         scores[longer[candidate[0]]['identity'],candidate[1]],draws=2000,seed=103)
    summary={'status':'COMPLETE_VARIABLE_DEPTH_'+a.split.upper(),'split':a.split,'rows':rows,
       'primary':primary,'secondary_exploratory':secondary,'longer_single_seed':pairs,
       'inventory_sha256':file_sha256(a.inventory),'analysis_sha256':file_sha256(__file__),
       'scope':'Fresh held-out groups; ordinary/fixed/mixture rerun with identical optimizer/topology/routing. Three primary seed contrasts use Bonferroni family intervals. All inference depths predeclared; no exit selection. Depth4 extrapolates training support. One-seed3B intervals condition on weights.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=f'''# Variable-depth pretraining: {a.split.upper()}

All natural-source targets and initial weights are paired. All models have
150,436,608 unique parameters. Fixed and mixture training execute exactly the
same estimated neural matmul work; mixture rank depths have probabilities
5/16, 6/16, 5/16 and mean two. Every listed depth is predeclared; no best exit
is selected. Depth4 is outside training support.

| Arm | Seed | Endpoint | Inference depth | Weighted NLL |
| --- | ---: | --- | ---: | ---: |
'''
    for r in rows:text+=f"| {r['arm']} | {r['seed']} | {r['endpoint']} | {r['depth']} | {r['metrics']['weighted_nll']:.6f} |\n"
    text+='\nPositive gain favors the candidate. Primary simultaneous intervals account for three contrasts.\n\n| Primary contrast | Mean gain | 95% seed interval | Family 95% interval | Family decision at 0.01 nats |\n| --- | ---: | --- | --- | --- |\n'
    for name,v in primary.items():
        lo,hi=v['interval_95'];flo,fhi=v['bonferroni_family_interval_95']
        text+=f"| {name} | {v['mean_gain_nats']:.6f} | [{lo:.6f}, {hi:.6f}] | [{flo:.6f}, {fhi:.6f}] | {v['family_decision']} |\n"
    text+='''
The mechanism, architecture and compute claims are separate. Better depth3
than depth1 alone does not establish an efficiency win, learned halting or an
interpretable reasoning algorithm. Student-t intervals have only two degrees
of freedom and assume approximately normal seed differences. This is one
frozen corpus/order. Domain scores, all secondary contrasts, explicit work and
any one-seed3B readouts are in the machine-readable report. Source-group
intervals cannot replace training-seed variation. Estimated training matmuls
include vocabulary projections, but not optimizer/communication/compiler cost.
The old TEST panel is consumed and was not used here. A new intervention after
this study needs another held-out panel.

[Full readout](summary.json) · [Protocol](../../docs/VARIABLE_DEPTH_PROTOCOL_20261023.md).
'''
    (a.output/'REPORT.md').write_text(text)
    print(json.dumps({'split':a.split,'primary':primary}),flush=True)


if __name__=='__main__':main()
