"""Descriptive matched-source TRAIN curves; no new model/data readouts."""
import json,time
from pathlib import Path
import numpy as np
from compute_trainium.training import file_sha256
root=Path('/run/user/1000/autonomous-compute/instance-nvme');out=Path('results/probability_workspace_20261029/untied-screen-v1')
models={}
for recipe,identity in [('tied','probability-fixed-one-v1'),('untied','probability-untied-screen-v1')]:
 plan=json.loads((root/identity/'plan.json').read_text())
 for arm in ('workspace','ordinary_data'):
  path=root/'runs'/plan['arms'][arm]['phase']/'updates.json';rows=json.loads(path.read_text())
  assert len(rows)==1530 and [r['update'] for r in rows]==list(range(1,1531))
  models[(recipe,arm)]=(rows,file_sha256(path))
reference=next(iter(models.values()))[0]
assert all([r['cumulative_real_targets'] for r in rows]==[r['cumulative_real_targets'] for r in reference] for rows,_ in models.values())
bounds=[(1,32),(33,256),(257,512),(513,768),(769,1024),(1025,1377),(1378,1530)]
records=[]
for recipe in ('tied','untied'):
 workspace=np.array([r['loss'] for r in models[(recipe,'workspace')][0]]);ordinary=np.array([r['loss'] for r in models[(recipe,'ordinary_data')][0]])
 for low,high in bounds:
  records.append(dict(recipe=recipe,update_start=low,update_end=high,workspace_mean_training_batch_nll=float(workspace[low-1:high].mean()),ordinary_mean_training_batch_nll=float(ordinary[low-1:high].mean()),ordinary_minus_workspace_mean_training_batch_nll=float((ordinary-workspace)[low-1:high].mean())))
r=dict(status='COMPLETE_DESCRIPTIVE_TRAINING_CURVE_COMPARISON',input_hashes={recipe+'-'+arm:sha for (recipe,arm),(_,sha) in models.items()},rows=records,optimizer_updates=0,model_reads=0,heldout_reads=0,assigned_chips=0,scope='Recorded pre-update TRAIN batch losses,exact same source/order through1530updates. Bins fixed before reading these curves. These are neither held-out performance nor supported within-checkpoint depth comparisons. Tied versus untied changes parameters;each has its own equal-data ordinary control. No uncertainty or efficiency claim from these curves.')
(out/'training-curves.json').write_text(json.dumps(r,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(9,4.5))
for recipe,color in [('tied','#a16207'),('untied','#2563eb')]:
 workspace=np.array([r['loss'] for r in models[(recipe,'workspace')][0]]);ordinary=np.array([r['loss'] for r in models[(recipe,'ordinary_data')][0]])
 smooth=np.convolve(ordinary-workspace,np.ones(64)/64,mode='valid');x=np.array([r['cumulative_real_targets']/1e6 for r in reference])[63:]
 ax.plot(x,smooth,color=color,label=recipe.title()+' tables')
ax.axhline(0,color='black',linewidth=.8);ax.axvspan(reference[1376]['cumulative_real_targets']/1e6,reference[-1]['cumulative_real_targets']/1e6,color='gray',alpha=.12,label='Final cooldown')
ax.set(xlabel='Natural training targets (millions)',ylabel='Ordinary minus workspace TRAIN NLL',title='Matched-source learning curves: 64-update rolling mean');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'training-curves.png',dpi=160);fig.savefig(out/'training-curves.svg');plt.close(fig)
print(json.dumps(r))
