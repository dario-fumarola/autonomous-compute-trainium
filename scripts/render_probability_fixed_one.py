"""Display the complete fixed-one comparison without changing its analysis."""
import argparse,json
from pathlib import Path

def main(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=json.loads((folder/'analysis.json').read_text())
    if a['status']!='COMPLETE_FIXED_ONE_ANALYSIS':raise AssertionError('Completed analysis required')
    labels=[('fixed-final-k1','Fixed one replay'),('sampled-final-k1','Sampled training: one replay'),('sampled-final-k2','Sampled training: two replays'),('ordinary_data-k0','Ordinary: equal data'),('ordinary_work-k0','Ordinary: equal work'),('uniform-k2','Constant: two mean additions')]
    pairs=[('ordinary_equal_work','Ordinary: equal work'),('ordinary_equal_data','Ordinary: equal data'),('sampled_same_exit','Sampled: same one-replay exit'),('sampled_deeper_exit','Sampled: two-replay exit'),('constant_two_additions','Constant: two mean additions')]
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for i,(key,label) in enumerate(labels):axes[0].plot(a['scores'][key]['broad_nll'],i,'o',color='#2367a4' if key=='fixed-final-k1' else '#6a6a6a')
    axes[0].set(yticks=range(len(labels)),yticklabels=[x[1] for x in labels],xlabel='Weighted development NLL (lower is better)',title='Final cooled endpoints');axes[0].invert_yaxis()
    for i,(key,label) in enumerate(pairs):
        x=a['comparisons'][key]['broad'];lo,hi=x['conditional_95_interval'];g=x['gain'];axes[1].errorbar(g,i,xerr=[[g-lo],[hi-g]],fmt='o',color='#2367a4',capsize=3)
    axes[1].axvline(0,color='black',linewidth=.8);axes[1].axvline(.01,color='#888888',linestyle='--',linewidth=.8)
    axes[1].set(yticks=range(len(pairs)),yticklabels=[x[1] for x in pairs],xlabel='Fixed-one gain, nats (positive is better)',title='Conditional 95% source-group intervals');axes[1].invert_yaxis()
    fig.suptitle('One new fit; reused development controls; no independent replication')
    fig.savefig(folder/'comparison.svg');fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
    text=['# Fixed one-replay optimization diagnostic','',
      'One new random-initialization fit is compared with the unchanged audited controls. The candidate and sampled-depth model have exactly the same200,190,896 real targets and190,988,262,121,144,320 nominal training matrix FLOPs. All final endpoints are cooled.','',
      '| Model | 60/20/20 NLL | Web | Math | Code |','|---|---:|---:|---:|---:|']
    for key,label in labels:
        s=a['scores'][key];text.append(f"| {label} | {s['broad_nll']:.6f} | {s['domains']['web']:.6f} | {s['domains']['math']:.6f} | {s['domains']['code']:.6f} |")
    text+=['','![All registered comparisons](comparison.png)','','| Comparator | Fixed-one gain | Conditional 95% interval |','|---|---:|---|']
    for key,label in pairs:
        x=a['comparisons'][key]['broad'];lo,hi=x['conditional_95_interval'];text.append(f"| {label} | {x['gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    text+=['','Only the fixed model’s one-replay exit is trained and supported. It cannot establish a within-checkpoint scaling curve. Source-group intervals condition on the trained weights;one seed and reused development sources do not establish training-seed robustness or confirmation.','',
      'The old constant control adds the mean embedding twice,not once,and is only a descriptive comparator. The semantic permutation preserves work/entropy but introduces distribution shift;its effect alone cannot establish reasoning. All arithmetic,code-name and exploratory intermediate-target results remain in[analysis.json](analysis.json). The nine code-name cases are not a multi-step reasoning benchmark.','',
      f"Repeated control readouts agree within{a['reused_readout_maximum_error']:.3g}nats under the unchanged2e-4 criterion. Independent CPU readout references are FP64;the Native predictor and fitted weights are unchanged. All320 protected source groups remain sealed.",'',
      'Inference requires2.2302× ordinary forward matrix work. That is a matrix-operation count,not measured cached decoding or a latency claim. Ordinary-work receives446,570,790 real targets. This screen uses one common preregistered LR,not an independently tuned conventional frontier.','',
      'New execution costs are in[execution-costs.json](execution-costs.json). Reused controls,failed attempts and numerical recovery retain their original ledgers in[the parent comparison](../../probability_workspace_20261028/evaluation-recovery-v2/REPORT.md). Reuse does not create a free new replication.','',
      'No automatic larger run,router or confirmation follows this result.']
    (folder/'REPORT.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);main(p.parse_args().folder)
