"""Paired held-out contrasts;source uncertainty is conditional,not seed evidence."""
import argparse,json
from pathlib import Path


def run(source,output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 a=json.loads((source/'analysis.json').read_text());c=a['contrasts']
 groups=[('Selected four predictions',[('latent_absolute_anchor_utility','Own raw − workspace'),('ordinary_minus_latent_anchor4','Ordinary − workspace'),('constant_minus_latent_anchor4','Trained constant − workspace')]),('Full 32-token continuation',[('ordinary_minus_latent_policy32','Ordinary − workspace'),('c0_minus_latent_policy32','C0 raw − workspace'),('constant_minus_latent_policy32','Trained constant − workspace')])]
 fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
 for ax,(title,rows) in zip(axes,groups,strict=True):
  for i,(key,label) in enumerate(rows):
   v=c[key];x=v['mean'];lo,hi=v['conditional_source_group_95_interval'];color='#1664b5' if x>=0 else '#b04430'
   ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color=color,capsize=4,markersize=5)
  ax.axvline(0,color='#555555',linewidth=1);ax.set_yticks(range(len(rows)),[v[1] for v in rows]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(title);ax.set_xlabel('NLL gain (nats); positive favors workspace')
 fig.suptitle('Uniform-NTP workspace discovery — conditional 95% source-group intervals')
 fig.text(.5,-.025,'One trained seed; 448 reused TUNE views / 116 groups. Intervals omit training-seed variation.',ha='center',fontsize=9)
 output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.output)
