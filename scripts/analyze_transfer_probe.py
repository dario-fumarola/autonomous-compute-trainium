"""Replay the frozen readout from retained features; publish aggregates only."""
import argparse
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from compute_trainium.metrics import DocumentScore,paired_group_interval
from compute_trainium.training import file_sha256
from scripts.run_transfer_probe import readout_fit


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);started=time.monotonic()
    source=json.loads((args.experiment/'report.json').read_text());examples=json.loads((args.data/'examples.json').read_text())
    if file_sha256(args.data/'examples.json')!=json.loads((args.experiment/'specification.json').read_text())['examples_sha256']:
        raise ValueError('Frozen examples changed')
    endpoint={m['model']:m for m in source['models']};scores={};max_error=0.
    for name in endpoint:
        state=torch.load(args.experiment/(name+'-features.pt'),map_location='cpu',weights_only=True)
        by_step={s:[] for s in [0,64,256,512]}
        for task in ['arc_challenge','race_high']:
            train=torch.tensor([i for i,d in enumerate(examples) if d['task']==task and d['split']=='train'])
            evaluation=torch.tensor([i for i,d in enumerate(examples) if d['task']==task and d['split']=='validation'])
            histories=[readout_fit(state['features'],state['original_logits'],state['labels'],train,evaluation,
                                   source['common_lr'],seed) for seed in [101,202,303]]
            for step in by_step:
                losses=np.mean([r[step]['losses'] for r in histories],axis=0)
                old=endpoint[name]['curves'][task][str(step)]['mean_nll']
                max_error=max(max_error,abs(float(losses.mean())-old))
                for i,value in zip(evaluation.tolist(),losses,strict=True):
                    d=examples[i];by_step[step].append(DocumentScore(task,d['id'],d['group'],float(value),1))
        scores[name]=by_step
    if max_error>1e-5:raise AssertionError('Independent replay differs from frozen readout')
    contrasts={}
    for seed in [11,12]:
        contrasts[str(seed)]={str(step):paired_group_interval(scores[f'scaffold_raw-seed{seed}'][step],
            scores[f'scaffold_trace-seed{seed}'][step],draws=2000,seed=17,
            weights={'arc_challenge':.5,'race_high':.5}) for step in [0,64,256,512]}
    model_table=[]
    for name,m in endpoint.items():
        model_table.append({'model':name,**{str(step):{'nll':statistics.mean(m['curves'][t][str(step)]['mean_nll'] for t in ['arc_challenge','race_high']),
            'accuracy':statistics.mean(m['curves'][t][str(step)]['mean_accuracy'] for t in ['arc_challenge','race_high'])}
            for step in [0,64,256,512]}})
    result={'status':'PASS_FROZEN_READOUT_REPLAY','maximum_replayed_mean_nll_error':max_error,
            'model_table':model_table,'contrasts_by_readout_updates':contrasts,
            'primary_step':512,'criterion':'0.01-nat lower bounds in both source orders for support; upper bounds below 0.01 rule out that size for this readout',
            'secondary_intervals':'Initial/intermediate snapshots are descriptive and non-simultaneous; do not replace primary step 512',
            'analysis_seconds':time.monotonic()-started,'source_report_sha256':file_sha256(args.experiment/'report.json'),
            'analysis_sha256':file_sha256(__file__)}
    (args.output/'replay-and-curves.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7,3.7),constrained_layout=True)
    for seed,color in [(11,'#3157a4'),(12,'#cf6a24')]:
        items=[contrasts[str(seed)][str(s)] for s in [0,64,256,512]]
        gains=[d['gain_nats'] for d in items]
        errors=[[g-d['interval_95'][0] for g,d in zip(gains,items)],
                [d['interval_95'][1]-g for g,d in zip(gains,items)]]
        ax.errorbar([0,64,256,512],gains,yerr=errors,fmt='o-',capsize=3,label=f'Source order {seed}',color=color)
    ax.axhline(0,color='black',lw=.8);ax.axhline(.01,color='gray',ls='--',lw=.8,label='Declared useful gain')
    ax.set(xlabel='Common linear-readout updates',ylabel='Raw − trace class NLL (nats/question)',
           title='Existing 1.7B checkpoints: initial gains and common adaptation')
    ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.2)
    fig.savefig(args.output/'readout-learning.png',dpi=180);fig.savefig(args.output/'readout-learning.svg');plt.close(fig)
    svg=args.output/'readout-learning.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    print(json.dumps({'maximum_replay_error':max_error,'contrasts':contrasts,'analysis_seconds':result['analysis_seconds']},indent=2))


if __name__=='__main__':main()
