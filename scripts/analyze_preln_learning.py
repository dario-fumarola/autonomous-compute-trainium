"""Paired three-seed recurrent curves and a separate fully annealed FLOP control."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compute_trainium.metrics import (DocumentScore, paired_group_interval,
                                      paired_three_seed_interval)
from compute_trainium.training import file_sha256


def interval(reference,candidate):
    return paired_three_seed_interval({str(seed):reference[seed]['metrics']['weighted_nll']-
                                      candidate[seed]['metrics']['weighted_nll'] for seed in [23,47,71]})


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    progress=json.loads((args.experiment/'progress.json').read_text());spec=progress['specification']
    ordinary=spec['reused_reference_runs'];recurrent=progress['records'];seeds=spec['seeds']
    def select(rows,step):return {r['seed']:r for r in rows if r['update']==step}
    curves=[];table=[]
    for step in [384,1152,3840]:
        a,b=select(ordinary,step),select(recurrent,step)
        if set(a)!=set(seeds) or set(b)!=set(seeds):continue
        if len({r['real_tokens'] for r in [*a.values(),*b.values()]})!=1:raise ValueError('Different paired source exposure')
        for seed in seeds:
            if a[seed]['initial_model_digest']!=b[seed]['initial_model_digest']:raise ValueError('Paired initial weights changed')
        value={'update':step,'real_targets':a[23]['real_tokens'],**interval(a,b),
               'ordinary_mean_nll':statistics.mean(r['metrics']['weighted_nll'] for r in a.values()),
               'recurrent_mean_nll':statistics.mean(r['metrics']['weighted_nll'] for r in b.values()),
               'ordinary_seed_nll':{str(s):a[s]['metrics']['weighted_nll'] for s in seeds},
               'recurrent_seed_nll':{str(s):b[s]['metrics']['weighted_nll'] for s in seeds},
               'schedule':'fully annealed' if step==3840 else 'stable trunk, descriptive',
               'domain_mean_nll':{arm:{d:statistics.mean(rows[s]['metrics']['domains'][d]['nll'] for s in seeds)
                    for d in ['web','math','code']} for arm,rows in [('ordinary',a),('recurrent',b)]}}
        curves.append(value)
        for arm,rows in [('ordinary',a),('recurrent',b)]:
            for seed,r in rows.items():
                table.append({'update':step,'real_targets':r['real_tokens'],'arm':arm,'seed':seed,
                              'weighted_nll':r['metrics']['weighted_nll'],
                              **{d+'_nll':r['metrics']['domains'][d]['nll'] for d in ['web','math','code']}})
    compute=None;a,b=select(ordinary,3840),select(recurrent,2269)
    if set(a)==set(b)==set(seeds):
        compute={**interval(a,b),'ordinary_real_targets':a[23]['real_tokens'],
                 'recurrent_real_targets':b[23]['real_tokens'],
                 'ordinary_training_matmul_flops':a[23]['training_matmul_flops'],
                 'recurrent_training_matmul_flops':b[23]['training_matmul_flops'],
                 'ordinary_mean_nll':statistics.mean(r['metrics']['weighted_nll'] for r in a.values()),
                 'recurrent_mean_nll':statistics.mean(r['metrics']['weighted_nll'] for r in b.values()),
                 'both_endpoints_fully_annealed':True}
    primary=next((c for c in curves if c['update']==3840),None)
    decision='PENDING_PRIMARY_ENDPOINT'
    if primary:
        lower,upper=primary['interval_95']
        decision='USEFUL_GAIN_SUPPORTED' if lower>.01 else ('USEFUL_GAIN_RULED_OUT_AT_THIS_EXPOSURE' if upper<.01 else 'INCONCLUSIVE')
    result={'status':'COMPLETE' if (args.experiment/'complete.json').exists() and primary else 'IN_PROGRESS',
            'curves':curves,'compute_matched':compute,'primary_decision':decision,
            'threshold_nats':.01,'selected_recurrent_lr':json.loads((args.experiment/'selected-lr.json').read_text())['selected_lr'],
            'ordinary_lr':spec['ordinary_selected_lr'],'parameters':spec['parameters'],
            'completed_recurrent_records':len(recurrent),'reference_controls_reused':True,
            'protocol_sha256':spec['protocol_sha256'],'specification_sha256':file_sha256(args.experiment/'specification.json'),
            'analysis_sha256':file_sha256(__file__),'scope':'Three paired initializations conditional on one frozen natural corpus/order; intermediate stable trunks are descriptive.'}
    if primary:
        root=args.experiment.parent;original=root.parent
        per_seed={}
        reference_scores=[];candidate_scores=[]
        for seed in seeds:
            ref=next(r for r in ordinary if r['seed']==seed and r['update']==3840)
            cand=next(r for r in recurrent if r['seed']==seed and r['update']==3840)
            aa=[DocumentScore(**v) for v in json.loads((original/'runs'/(ref['run']+'-dev')/'document-scores.json').read_text())]
            bb=[DocumentScore(**v) for v in json.loads((root/'runs'/(cand['run']+'-dev')/'document-scores.json').read_text())]
            per_seed[str(seed)]=paired_group_interval(aa,bb,draws=2000,seed=23)
            reference_scores.append({s.document:s for s in aa});candidate_scores.append({s.document:s for s in bb})
        def averaged(sets):
            rows=[]
            for key,first in sets[0].items():
                values=[s[key] for s in sets]
                if any((s.domain,s.group,s.tokens)!=(first.domain,first.group,first.tokens) for s in values):raise ValueError('Source identity changed across seeds')
                rows.append(DocumentScore(first.domain,key,first.group,statistics.mean(s.nll_sum for s in values),first.tokens))
            return rows
        result['source_group_uncertainty']={'per_seed':per_seed,
            'fixed_seed_mean':paired_group_interval(averaged(reference_scores),averaged(candidate_scores),draws=2000,seed=23),
            'scope':'Resampled source groups conditional on fitted checkpoints; not joint seed/data uncertainty or an ensemble model.'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if table:
        with (args.output/'learning-curve.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    if curves:
        x=[c['real_targets']/1e9 for c in curves]
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for arm,color in [('ordinary','#3157a4'),('recurrent','#cf6a24')]:
            axes[0].plot(x,[c[arm+'_mean_nll'] for c in curves],marker='o',color=color,label=arm)
        axes[0].set(xscale='log',xlabel='Real training targets (billions)',ylabel='Held-out weighted NLL')
        axes[0].set_xticks([.1,.3,1.0],['0.1','0.3','1.0']);axes[0].legend(frameon=False)
        gains=[c['mean_gain_nats'] for c in curves]
        errors=[[c['mean_gain_nats']-c['interval_95'][0] for c in curves],
                [c['interval_95'][1]-c['mean_gain_nats'] for c in curves]]
        axes[1].errorbar(x,gains,yerr=errors,fmt='o-',capsize=4,color='#cf6a24')
        axes[1].axhline(0,color='black',lw=.8);axes[1].axhline(.01,color='gray',ls='--',lw=.8,label='Declared useful gain')
        axes[1].set(xscale='log',xlabel='Real training targets (billions)',ylabel='Ordinary − recurrent NLL')
        axes[1].set_xticks([.1,.3,1.0],['0.1','0.3','1.0']);axes[1].legend(frameon=False,fontsize=8)
        for ax in axes:ax.grid(alpha=.2);ax.margins(x=.08)
        fig.suptitle('Fixed recurrence: three paired initialization seeds')
        fig.savefig(args.output/'learning-curve.png',dpi=180);fig.savefig(args.output/'learning-curve.svg');plt.close(fig)
        svg=args.output/'learning-curve.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    print(json.dumps({'status':result['status'],'completed_records':len(recurrent),
                     'paired_milestones':len(curves),'decision':decision,'latest':curves[-1] if curves else None},indent=2))


if __name__=='__main__':main()
