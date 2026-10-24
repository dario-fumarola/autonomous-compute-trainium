"""Exploratory first-pair curves; frozen three-seed analysis stays unchanged."""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compute_trainium.metrics import DocumentScore,paired_group_interval,summarize
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[];documents={};provenance={};initial=set();source=set();dev=set()
    for arm,updates in [('ordinary',[384,1152,3840,6496]),('mixture',[384,1152,3840])]:
        depth=1 if arm=='ordinary' else 2
        for update in updates:
            name=f'vdepth-v1-{arm}-s103-u{update}'
            model=a.root/'runs'/name/'report.json';readout=a.root/'runs'/f'{name}-dev-r{depth}'
            train=json.loads(model.read_text());scored=json.loads((readout/'report.json').read_text())
            if train['status']!='PASS_PACKED_TRAINING_PHASE' or scored['status']!='PASS_DEPTH_MATRIX_READOUT':
                raise ValueError('Incomplete fitted model or scoring phase')
            if not scored['weights_unchanged_all_replicas'] or not train['all_replica_model_optimizer_states_equal']:
                raise ValueError('Unverified fitted/readout state')
            if train['parameters']!=150436608:raise ValueError('Parameter mismatch')
            initial.add(train['initial_model_digest']);source.add(train['signature']['corpus_sha256']);dev.add(scored['corpus_sha256'])
            values=[DocumentScore(**d) for d in json.loads((readout/'document-scores.json').read_text())]
            metrics=summarize(values)
            if abs(metrics['weighted_nll']-scored['metrics']['weighted_nll'])>1e-10:
                raise ValueError('Document totals do not reproduce readout')
            rows.append({'arm':arm,'seed':103,'update':update,'depth':depth,
                         'targets':train['total_real_targets'],'nll':metrics['weighted_nll'],
                         'training_update_matmul_flops':train['total_training_matmul_flops'],
                         'schedule':'stable' if update in [384,1152] else 'annealed'})
            documents[arm,update]=values
            provenance[name]={'training_report_sha256':file_sha256(model),
                              'readout_report_sha256':file_sha256(readout/'report.json'),
                              'private_document_scores_sha256':file_sha256(readout/'document-scores.json')}
    if len(initial)!=1 or len(source)!=1 or len(dev)!=1:raise ValueError('Initial weights or source panels differ')
    comparisons=[]
    for update in [384,1152,3840]:
        pair=paired_group_interval(documents['ordinary',update],documents['mixture',update],draws=2000,seed=103)
        labels={384:'100M, equal data',1152:'301M, equal data',3840:'1.005B, equal data'}
        comparisons.append({'label':labels[update],'update':update,**pair})
    comparisons.append({'label':'Matched update matmuls','update':None,**paired_group_interval(
                        documents['ordinary',6496],documents['mixture',3840],draws=2000,seed=103)})
    lookup={(r['arm'],r['update']):r for r in rows}
    shortfall=1-lookup['ordinary',6496]['training_update_matmul_flops']/lookup['mixture',3840]['training_update_matmul_flops']
    if not 0<=shortfall<.00014:raise ValueError('Incorrect compute control')
    result={'status':'EXPLORATORY_FIRST_PAIR_CURVE','created_unix':time.time(),'rows':rows,
            'source_group_intervals':comparisons,'compute_rounding_shortfall_fraction':shortfall,
            'provenance':provenance,'script_sha256':file_sha256(__file__),
            'scope':'Seed103 only. Same targets/initialization/parameters; independently annealed endpoints. Group-bootstrap95% intervals condition on these fitted weights and are descriptive, without multiplicity adjustment or a seed-level hypothesis decision. No new fit or TEST scoring; frozen primary analysis is unchanged.'}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.8),layout='constrained',gridspec_kw={'width_ratios':[1,1.25]})
    colors={'ordinary':'#b65c26','mixture':'#167684'}
    for arm in ['ordinary','mixture']:
        values=[lookup[arm,u] for u in [384,1152,3840]]
        axes[0].plot([r['targets']/1e9 for r in values],[r['nll'] for r in values],
                     color=colors[arm],linestyle=':',linewidth=1.5,
                     label='ordinary (1 pass)' if arm=='ordinary' else 'mixed training, evaluated at 2 passes')
        for row in values:
            axes[0].scatter(row['targets']/1e9,row['nll'],color=colors[arm],
                            marker='o' if row['schedule']=='stable' else 's',s=36,zorder=3)
    control=lookup['ordinary',6496]
    axes[0].scatter(control['targets']/1e9,control['nll'],marker='D',facecolors='none',
                    edgecolors=colors['ordinary'],s=65,label='ordinary, compute control')
    axes[0].set(title='Learning curve: one paired seed',xlabel='Real source targets (billions)',
                ylabel='Fresh DEV NLL (lower is better)',xlim=(.02,1.80))
    axes[0].legend(frameon=False,fontsize=8)
    for y,row in enumerate(comparisons):
        lo,hi=row['interval_95'];mean=row['gain_nats']
        axes[1].errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],color='#167684',fmt='o',capsize=3)
        axes[1].text(.98,y+.19,f'{mean:+.5f} [{lo:+.5f}, {hi:+.5f}]',
                     transform=axes[1].get_yaxis_transform(),ha='right',fontsize=7.5)
    axes[1].axvline(0,color='#a4aab2',linewidth=.8)
    axes[1].axvline(.01,color='#7c8390',linestyle='--',linewidth=.9)
    axes[1].set(yticks=range(4),yticklabels=[r['label'] for r in comparisons],
                xlabel='Ordinary NLL − mixture NLL (nats)',title='Candidate gain and source-group uncertainty',ylim=(3.7,-.65))
    axes[1].text(.99,.02,'Positive favors mixture; dashed line: +0.01',
                 transform=axes[1].transAxes,ha='right',fontsize=7.5)
    fig.suptitle('Teacher-free natural-text pretraining · 150M parameters · seed 103',fontsize=12)
    fig.supxlabel('Circles: stable LR. Squares/diamond: completed decay. The 1.7B control branches from 904M stable, not the 1B endpoint.\nIntervals: source-group bootstrap conditional on fitted weights; training-seed uncertainty remains pending.',fontsize=8)
    for suffix in ['png','svg','pdf']:fig.savefig(a.output/f'first-pair-learning.{suffix}')
    plt.close(fig)
    lines=['# First paired learning curve — exploratory DEV',
           '',result['scope'],'',
           '| Comparison | Mixture gain (nats) | Conditional source-group95% interval |',
           '| --- | ---: | --- |']
    for row in comparisons:
        lo,hi=row['interval_95'];lines.append(f"| {row['label']} | {row['gain_nats']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','The final data-matched gain and the compute comparison answer different questions.',
            'The compute control uses1.700B ordinary targets versus1.005B mixture targets.',
            'Earlier points use stable learning rates; endpoints complete their own decay.',
            'No source contents, document/group identifiers or weights are released.',
            '','[Figure](first-pair-learning.png) · [Numerical readout](summary.json).']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'conditional_intervals':comparisons}),flush=True)


if __name__=='__main__':main()
