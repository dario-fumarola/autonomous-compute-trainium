"""Plot recorded global TRAIN metrics,without relabeling them as DEV results."""
import argparse,csv
from pathlib import Path


def run(a):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors=dict(latent='#1463c5',constant='#d28110',ordinary='#333333')
 figure,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
 for arm in ['latent','constant','ordinary']:
  path=a.source/(arm+'-recorded-global-training.csv')
  rows=list(csv.DictReader(path.open()))
  for axis,field,title in zip(axes,['raw159_nll','anchor_nll'],['Ordinary natural targets','Four selected targets'],strict=True):
   groups=[rows[k:k+50] for k in range(0,len(rows),50)]
   x=[64*sum(int(v['update']) for v in g)/len(g) for g in groups]
   y=[sum(float(v[field]) for v in g)/len(g) for g in groups]
   axis.plot(x,y,label=arm,color=colors[arm],linewidth=1.7)
   axis.set(title=title,xlabel='Source views processed',ylabel='Mean training NLL')
   axis.grid(alpha=.2)
 axes[0].legend(frameon=False)
 figure.suptitle('Recorded training trajectories — means over 50 updates')
 figure.text(.5,-.035,'Different source views at each update; shared order across arms. These curves are not held-out evaluation.',ha='center',fontsize=9)
 a.output.parent.mkdir(parents=True,exist_ok=True);figure.savefig(a.output,dpi=160,bbox_inches='tight');plt.close(figure)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
