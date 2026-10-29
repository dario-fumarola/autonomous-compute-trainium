"""Render completed, frozen comparisons; never select or recompute outcomes."""
import argparse,json
from pathlib import Path

def main(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=json.loads((folder/'plan.json').read_text());a=json.loads((folder/'analysis.json').read_text())
    if a['status']!='COMPLETE_PROBABILITY_SCREEN_ANALYSIS':raise AssertionError('Completed frozen analysis required')
    scores=a['scores'];fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for stage,color,label in [('early','#c56724','64.4M targets (uncooled)'),('final','#2367a4','200.2M targets (cooled)')]:
        axes[0].plot([0,1,2],[scores[f'workspace-{stage}-k{k}']['broad_nll'] for k in (0,1,2)],'o-',color=color,label=label)
        ordinary=scores[f'ordinary-data-{stage}-k0']['broad_nll']
        axes[0].axhline(ordinary,color=color,linestyle=':',alpha=.7,label=f'Ordinary, {stage}, equal data')
    axes[0].set(xlabel='Supported extra passes, same workspace checkpoint',ylabel='Weighted held-out NLL (lower is better)',xticks=[0,1,2],title='Does additional supported computation help?')
    axes[0].legend(fontsize=8)
    names=[('ordinary_equal_work','Ordinary: equal training work'),('ordinary_equal_data','Ordinary: equal data'),('constant_proposal_control','Trained constant proposal'),('final_extra_2','Same checkpoint: one extra pass'),('final_extra_total','Same checkpoint: zero extra passes')]
    for i,(name,label) in enumerate(names):
        r=a['comparisons'][name]['broad'];lo,hi=r['conditional_95_interval'];g=r['gain']
        axes[1].errorbar(g,i,xerr=[[max(0,g-lo)],[max(0,hi-g)]],fmt='o',color='#2367a4',capsize=3)
    axes[1].axvline(0,color='black',linewidth=.8);axes[1].axvline(.01,color='#888888',linestyle='--',linewidth=.8)
    axes[1].set(yticks=range(len(names)),yticklabels=[x[1] for x in names],xlabel='Gain of workspace deepest exit, nats (positive is better)',title='Final comparisons: conditional 95% intervals')
    axes[1].invert_yaxis();fig.suptitle('One training seed; development sources; not confirmation',fontsize=12)
    fig.savefig(folder/'comparison.svg');fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
    rows=[]
    entries=[('Workspace, zero replay','workspace-final-k0'),('Workspace, one replay','workspace-final-k1'),('Workspace, two replays','workspace-final-k2'),('Ordinary, equal data','ordinary-data-final-k0'),('Ordinary, equal work','ordinary-work-final-k0'),('Uniform proposal, equal data','uniform-final-k2')]
    for label,key in entries:
        s=scores[key];rows.append(f"| {label} | {s['broad_nll']:.6f} | {s['domains']['web']:.6f} | {s['domains']['math']:.6f} | {s['domains']['code']:.6f} |")
    lines=['# Probability replay: completed development screen','',
        'All four fits, final cooldowns, mature numerical audits and registered evaluation are complete. One random initialization and one common preregistered learning rate; this is not an independently tuned conventional frontier or confirmation.','',
        '| Predictor | 60/20/20 NLL | Web | Math | Code |','|---|---:|---:|---:|---:|',*rows,'',
        '![Registered comparisons](comparison.png)','',
        'Positive paired gains below favor the deepest workspace exit. Intervals reflect evaluation-source variation conditional on these trained weights, not uncertainty across training runs.','',
        '| Comparison | Gain, nats | Conditional 95% interval |','|---|---:|---|']
    for name,label in names:
        r=a['comparisons'][name]['broad'];lo,hi=r['conditional_95_interval'];lines.append(f"| {label} | {r['gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','The practical screening threshold is0.01nats with a positive conditional lower bound. These decisions do not establish reasoning, autonomous allocation, training-seed robustness or serving efficiency.','',
        'Workspace/equal-data/uniform arms each use200,190,896 real targets;ordinary-work uses446,570,790. All final endpoints independently cool to0.1peak learning rate. Earlier64.4M checkpoints are uncooled descriptive points. All320 protected source groups remain unscored.','',
        'The deepest workspace readout costs3.4604× ordinary forward matrix work;the sampled training average is2.2302×. These are dense-matrix estimates,not cached-decoding or latency measurements. The uniform proposal is cheaper and does not execute a proposal body.','',
        'Full dependency results, all domain signs, the proposal-content intervention, and every registered contrast remain in[analysis.json](analysis.json). Only9code-name cases fit the frozen context;those are not a code-reasoning benchmark. A corruption effect alone is not proof of content-based reasoning.','',
        'The original complete workspace fit is reused unchanged after correcting an OrderedDict/dict representation mismatch in the CPU restoration hash. Both checkpoint digests reproduce exactly using the original Native serializer. No numerical tolerance changed. Original stopped-controller costs remain in[the original ledger](../screen-v1/execution-costs.json);recovery work is separate in[execution-costs.json](execution-costs.json).','',
        'No automatic larger fit,router or confirmation is admitted by this screen.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);main(p.parse_args().folder)
