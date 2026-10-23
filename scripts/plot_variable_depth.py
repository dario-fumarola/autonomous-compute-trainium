"""Standalone scientific figures from completed, predeclared depth readouts."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary=json.loads(a.summary.read_text());rows=summary['rows'];a.output.mkdir(parents=True,exist_ok=True)
    colors={'ordinary':'#243b53','fixed':'#bc4b51','mixture':'#007f86'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.3),layout='constrained')
    table=[]
    for arm in ['ordinary','fixed','mixture']:
        depth=1 if arm=='ordinary' else 2;points=[]
        for update,style in [(384,'stable'),(1152,'stable'),(3840,'annealed')]:
            scores=[];targets=None
            for seed in [103,211,307]:
                path=a.root/'runs'/f'vdepth-v1-{arm}-s{seed}-u{update}-dev-r{depth}'/'report.json'
                d=json.loads(path.read_text());scores.append(d['metrics']['weighted_nll']);targets=d['training_real_tokens']
            points.append((targets/1e9,statistics.mean(scores),style));table.append({'arm':arm,'update':update,'targets':targets,'scores':scores,'schedule':style})
        axes[0].plot([v[0] for v in points[:2]],[v[1] for v in points[:2]],color=colors[arm],marker='o',label=arm)
        axes[0].plot([points[1][0],points[2][0]],[points[1][1],points[2][1]],color=colors[arm],linestyle=':')
        axes[0].scatter([points[2][0]],[points[2][1]],color=colors[arm],marker='D',s=40)
    axes[0].set(xlabel='Real source targets (billions)',ylabel='Fresh DEV weighted NLL',title='Learning curves: three-seed means')
    axes[0].legend(frameon=False);axes[0].text(.02,.02,'Circles: stable LR; diamonds: completed decay',transform=axes[0].transAxes,fontsize=8)
    for arm in ['fixed','mixture']:
        ds=[];means=[];halves=[]
        for depth in [1,2,3,4]:
            values=[r['metrics']['weighted_nll'] for r in rows if r['arm']==arm and r['endpoint']=='1B' and r['depth']==depth]
            assert len(values)==3
            ds.append(depth);means.append(statistics.mean(values));halves.append(4.302652729911275*statistics.stdev(values)/3**.5)
        axes[1].errorbar(ds,means,yerr=halves,color=colors[arm],marker='o',capsize=3,label=arm,linewidth=1.5)
    ordinary=statistics.mean(r['metrics']['weighted_nll'] for r in rows if r['arm']=='ordinary' and r['endpoint']=='1B')
    control=statistics.mean(r['metrics']['weighted_nll'] for r in rows if r['arm']=='ordinary' and r['endpoint']=='matched')
    axes[1].axhline(ordinary,color=colors['ordinary'],label='ordinary, 1B',linewidth=1.2)
    axes[1].axhline(control,color=colors['ordinary'],label='ordinary, matched train matmuls',linestyle='--',linewidth=1.2)
    axes[1].axvspan(3.5,4.4,color='#eef0f3');axes[1].set(xlabel='Inference core applications',ylabel=f"Fresh {summary['split'].upper()} weighted NLL",
        title='1B endpoints: same weights, different depths',xticks=[1,2,3,4],xlim=(.85,4.35))
    axes[1].legend(frameon=False,fontsize=8);axes[1].text(.02,.02,'Intervals: seed95%, df=2. Fixed trains at2; mixture at1/2/3.\nDepth4 extrapolates; ordinary matched control sees1.7B targets.',transform=axes[1].transAxes,fontsize=7.5)
    fig.suptitle('Variable-depth pretraining on natural text — 150M parameters',fontsize=13)
    for suffix in ['png','svg','pdf']:fig.savefig(a.output/f"learning-and-depth-{summary['split']}.{suffix}")
    (a.output/'learning-curve-data.json').write_text(json.dumps(table,indent=2)+'\n');plt.close(fig)


if __name__=='__main__':main()
