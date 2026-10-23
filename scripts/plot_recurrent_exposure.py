"""Single-seed learning curve with annealed branches marked separately."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    complete=json.loads((a.root/'stable-continuation-v2/report.json').read_text())
    if complete['status']!='COMPLETE_SINGLE_PAIR_3B_CONTINUATION':raise ValueError('Finish the fixed 3B pair first')
    original=list(csv.DictReader(Path('results/recurrent_learning_20261023/learning-curve.csv').open()))
    rows=[]
    for r in original:
        if int(r['seed'])==23:
            rows.append({'arm':r['arm'],'targets':int(r['real_targets']),'nll':float(r['weighted_nll']),
                'stage':'annealed_1b' if int(r['update'])==3840 else 'stable'})
    for r in complete['records']:
        rows.append({'arm':r['arm'],'targets':r['real_tokens'],'nll':r['metrics']['weighted_nll'],
            'stage':'annealed_3b' if r['update']==11520 else 'stable'})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    colors={'ordinary':'#2869a6','recurrent':'#cc6b25'}
    for arm,label in [('ordinary','Ordinary'),('recurrent','Shared two-pass')]:
        stable=sorted([r for r in rows if r['arm']==arm and r['stage']=='stable'],key=lambda r:r['targets'])
        axes[0].plot([r['targets']/1e9 for r in stable],[r['nll'] for r in stable],'o-',color=colors[arm],label=label)
        for stage,marker in [('annealed_1b','D'),('annealed_3b','*')]:
            end=next(r for r in rows if r['arm']==arm and r['stage']==stage)
            axes[0].scatter(end['targets']/1e9,end['nll'],marker=marker,s=65 if marker=='D' else 130,color=colors[arm],zorder=3)
            if stage=='annealed_3b':
                axes[0].plot([stable[-1]['targets']/1e9,end['targets']/1e9],[stable[-1]['nll'],end['nll']],':',color=colors[arm])
    axes[0].set(xlabel='Real source targets (billions)',ylabel='DEV weighted NLL (nats / real token)',title='One paired seed: 23')
    axes[0].legend();axes[0].grid(alpha=.15)
    grouped={}
    for row in rows:grouped.setdefault((row['targets'],row['stage']),{})[row['arm']]=row['nll']
    stable=[]
    for (targets,stage),values in sorted(grouped.items()):
        if set(values)!={'ordinary','recurrent'}:raise ValueError('Learning-curve targets are not paired')
        gain=values['ordinary']-values['recurrent']
        if stage=='stable':stable.append((targets/1e9,gain))
        else:axes[1].scatter(targets/1e9,gain,marker='D' if stage=='annealed_1b' else '*',s=65 if stage=='annealed_1b' else 130,color='#454545',zorder=3)
    axes[1].plot([x for x,y in stable],[y for x,y in stable],'o-',color='#454545',label='Stable checkpoints')
    axes[1].axhline(0,color='#555555',linewidth=.8)
    axes[1].axhline(.01,color='#2b8755',linestyle='--',linewidth=1,label='Declared useful gain: 0.01')
    axes[1].set(xlabel='Real source targets (billions)',ylabel='Ordinary NLL − shared NLL',title='Positive favors shared recurrence')
    axes[1].legend();axes[1].grid(alpha=.15)
    fig.suptitle('Shared recurrence with additional natural-text exposure',fontsize=13)
    fig.supxlabel('Diamonds: separately annealed 1B branch. Stars: annealed 3B endpoint. No seed interval from this one pair.',fontsize=9)
    for suffix in ['png','svg','pdf']:fig.savefig(a.output/('learning-curve-seed23.'+suffix),dpi=180)
    (a.output/'learning-curve-seed23.json').write_text(json.dumps({'rows':rows,'scope':'One paired seed, fixed DEV panel. Stable points and separately annealed endpoints are distinct training branches; lines are descriptive, not evidence of convergence. No TEST selection.'},indent=2)+'\n')


if __name__=='__main__':main()
