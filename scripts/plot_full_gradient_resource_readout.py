"""Standalone completed-DEV readout;no new scoring or outcome selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compute_trainium.training import file_sha256


def main():
    source=Path('results/future_span_main_20261025/primitive-v2/report.json')
    work=Path('results/future_span_work_controls_20261025/report.json')
    original=json.loads(source.read_text());controls=json.loads(work.read_text())
    if (original['status']!='COMPLETE_BOUNDED_FUTURE_SPAN_OBJECTIVE_PILOT'
            or controls['status']!='COMPLETE_WITHIN_FAMILY_WORK_MATCHED_CONTROLS'):
        raise ValueError('Complete original studies only')
    seeds=[5231,7283]
    def pick(rows,cell,depth):
        return [next(r for r in rows if r['seed']==s and r['cell']==cell and r['depth']==depth
                     and r.get('readout','cooled1b')=='cooled1b') for s in seeds]
    one=pick(original['scores'],'one-ntp',1)
    recurrent={d:pick(original['scores'],'recurrent-ntp',d) for d in [1,2,3]}
    longer=pick(controls['scores'],'one-ntp',1)
    nll=lambda rows:np.array([r['metrics']['weighted_nll'] for r in rows])
    base=one[0]['estimated_training_dense_matrix_ops']
    x=np.array([1.,longer[0]['estimated_training_dense_matrix_ops']/base])
    rx=recurrent[2][0]['estimated_training_dense_matrix_ops']/base
    blue,orange='#225ea8','#d95f0e'
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.7))
    ax=axes[0]
    ax.plot(x,[nll(one).mean(),nll(longer).mean()],color=blue,marker='o',label='One-pass NTP')
    for j in range(2):ax.plot(x,[nll(one)[j],nll(longer)[j]],color=blue,alpha=.25,linewidth=.8)
    ax.scatter([rx], [nll(recurrent[2]).mean()],color=orange,s=65,marker='D',label='Recurrent NTP, depth 2',zorder=4)
    ax.scatter([rx,rx],nll(recurrent[2]),color=orange,s=15,alpha=.6)
    ax.set(xlabel='Complete nominal training matrix work / one-pass 1B',ylabel='Consumed DEV NLL ↓',
           title='Equal-data gain; matched-work loss',xlim=(.985,1.255))
    ax.legend(frameon=False,fontsize=8)
    ax=axes[1]
    values=np.array([nll(recurrent[d]) for d in [1,2,3]])
    ax.plot([1,2,3],values.mean(1),color=orange,marker='o',label='Same recurrent weights')
    for j in range(2):ax.plot([1,2,3],values[:,j],color=orange,alpha=.25,linewidth=.8)
    ax.axhline(nll(one).mean(),color=blue,linestyle=':',label='Trained one-pass, equal data')
    ax.axhline(nll(longer).mean(),color=blue,linestyle='--',label='One-pass, matched training work')
    ax.set(xlabel='Supported inference depth',ylabel='Consumed DEV NLL ↓',
           title='Additional depth helps these weights',xticks=[1,2,3])
    ax.legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.17)
    fig.suptitle('Plain NTP, 141.96M deployed parameters, two paired development seeds',fontsize=12)
    fig.text(.5,.025,'Every training endpoint completes its own decay. Thin lines show individual seeds, not confidence intervals.\n'
        'Primary inference exit remains 2. Depth 3 is diagnostic. No fresh confirmation or hardware-cost matching.',
        ha='center',fontsize=8)
    fig.subplots_adjust(bottom=.23,wspace=.30,top=.82)
    out=Path('results/future_span_work_controls_20261025/figures');out.mkdir(exist_ok=True)
    for extension in ['png','svg','pdf']:fig.savefig(out/('completed-resource-readout.'+extension),dpi=180)
    plt.close(fig)
    (out/'provenance.json').write_text(json.dumps({'source_sha256':file_sha256(source),
        'control_sha256':file_sha256(work),'script_sha256':file_sha256(__file__),
        'seeds':seeds,'new_model_forwards':0,'optimizer_updates':0,'assigned_chips':0},indent=2)+'\n')


if __name__=='__main__':main()
