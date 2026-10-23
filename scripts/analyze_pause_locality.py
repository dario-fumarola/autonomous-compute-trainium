"""Aggregate the prospectively fixed pause-distance bins; exploratory only."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compute_trainium.metrics import paired_three_seed_interval
from compute_trainium.training import file_sha256
from scripts.diagnose_pause_locality import BIN_NAMES


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=True)
    weights={'web':.6,'math':.2,'code':.2};by_seed={};source_hashes={}
    for seed in [23,47,71]:
        files=[args.input/f'natural-curve-v1-{arm}-s{seed}-u3840-scores.json' for arm in ['ordinary','early']]
        rows=[{d['document']:d for d in json.loads(path.read_text())} for path in files]
        for path in files:source_hashes[path.name]=file_sha256(path)
        a,b=rows
        if a.keys()!=b.keys():raise ValueError('Document pairing changed')
        totals={d:0 for d in weights};counts={d:[0]*6 for d in weights};deltas={d:[0.]*6 for d in weights}
        for key,left in a.items():
            right=b[key]
            if (left['domain'],left['group'],left['tokens_by_bin'])!=(right['domain'],right['group'],right['tokens_by_bin']):
                raise ValueError('Source target pairing changed')
            domain=left['domain'];totals[domain]+=sum(left['tokens_by_bin'])
            for i in range(6):
                counts[domain][i]+=left['tokens_by_bin'][i]
                deltas[domain][i]+=left['nll_sum_by_bin'][i]-right['nll_sum_by_bin'][i]
        bins=[]
        for i,name in enumerate(BIN_NAMES):
            mass=sum(weights[d]*counts[d][i]/totals[d] for d in weights)
            contribution=sum(weights[d]*deltas[d][i]/totals[d] for d in weights)
            bins.append({'bin':name,'mixture_target_mass':mass,'conditional_gain_nats':contribution/mass,
                         'contribution_to_overall_gain':contribution,
                         'domain_gains':{d:deltas[d][i]/counts[d][i] for d in weights},
                         'domain_targets':{d:counts[d][i] for d in weights}})
        overall=sum(weights[d]*sum(deltas[d])/totals[d] for d in weights)
        assert abs(sum(v['contribution_to_overall_gain'] for v in bins)-overall)<1e-12
        assert abs(sum(v['mixture_target_mass'] for v in bins)-1)<1e-12
        by_seed[str(seed)]={'bins':bins,'overall_gain_nats':overall,'domain_targets':totals}
    summary=[]
    for i,name in enumerate(BIN_NAMES):
        interval=paired_three_seed_interval({s:v['bins'][i]['conditional_gain_nats'] for s,v in by_seed.items()})
        summary.append({'bin':name,**interval,
            'mixture_target_mass':by_seed['23']['bins'][i]['mixture_target_mass'],
            'mean_contribution_to_overall_gain':statistics.mean(v['bins'][i]['contribution_to_overall_gain'] for v in by_seed.values()),
            'domain_mean_gains':{d:statistics.mean(v['bins'][i]['domain_gains'][d] for v in by_seed.values()) for d in weights}})
    original=json.loads((args.input/'report.json').read_text())
    result={'status':'COMPLETE_EXPLORATORY_PAUSE_LOCALITY','bins':summary,'seeds':by_seed,
        'mean_overall_cpu_gain_nats':statistics.mean(v['overall_gain_nats'] for v in by_seed.values()),
        'cpu_minus_original_native_nll':[{k:v for k,v in row.items() if k in ['seed','arm','cpu_minus_original_native_weighted_nll']} for row in original['models']],
        'source_sha256':source_hashes,
        'interpretation':'Conditional bin gains use the domain mixture conditioned on that bin. Weighted contributions reconstruct overall gain exactly. Six non-simultaneous paired-seed intervals are exploratory; no new success criterion.',
        'scope':'Different trained ordinary/pause weights on the same targets, not an inference deletion intervention.'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    labels=['Before first\nblock','Next token\n(offset 0)','Offsets\n1–7','Offsets\n8–31','Offsets\n32–63','Offsets\n64–127']
    means=[d['mean_gain_nats'] for d in summary]
    errors=[[d['mean_gain_nats']-d['interval_95'][0] for d in summary],
            [d['interval_95'][1]-d['mean_gain_nats'] for d in summary]]
    fig,ax=plt.subplots(figsize=(8,3.8),constrained_layout=True)
    ax.errorbar(range(6),means,yerr=errors,fmt='o',capsize=4,color='#3157a4')
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(6),labels)
    ax.set_ylabel('Ordinary − pause NLL (nats/target)')
    ax.set_title('Existing 1B checkpoints: pause gain by source position')
    ax.grid(axis='y',alpha=.2)
    fig.savefig(args.output/'pause-locality.png',dpi=180);fig.savefig(args.output/'pause-locality.svg');plt.close(fig)
    svg=args.output/'pause-locality.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({'overall_gain':result['mean_overall_cpu_gain_nats'],
        'bins':[{k:v for k,v in d.items() if k in ['bin','mean_gain_nats','interval_95']} for d in summary]},indent=2))


if __name__=='__main__':main()
