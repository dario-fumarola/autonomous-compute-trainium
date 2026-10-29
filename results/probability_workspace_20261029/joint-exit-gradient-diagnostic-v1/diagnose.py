"""Bounded zero-update TRAIN gradient compatibility, not an exit quality claim."""
import json,time,resource
from pathlib import Path
import torch
from compute_trainium.training import file_sha256
from compute_trainium.probability_model_factory import probability_model,gradient_families
from compute_trainium.probability_workspace_data import ProbabilitySources
from compute_trainium.probability_objective import reference_loss_sum
ROOT=Path('/run/user/1000/autonomous-compute/instance-nvme');OUT=Path(__file__).resolve().parent
start=time.time();torch.set_num_threads(8)
parent=ROOT/'probability-untied-screen-v1/plan.json';p=json.loads(parent.read_text())
old=ROOT/'runs/probability-untied-screen-v1-workspace/PRIVATE-u1530.pt'
new=ROOT/'runs/probability-joint-exit-warm-v1-warm/PRIVATE-u103.pt'
plan=dict(status='REGISTERED_TRAIN_ONLY_GRADIENT_DIAGNOSTIC',registered_unix=start,window_start=1000000,windows=2,
 models=['initial','final_only_u1530','joint_disposable_u103'],checkpoint_hashes={str(f.relative_to(ROOT)):file_sha256(f) for f in (old,new)},
 parent_plan_sha256=file_sha256(parent),seconds_limit=600,memory_limit_gib=48,source_split='TRAIN audit windows outside fitted prefixes',
 scope='No updates,no heldout read,no LR/exit-weight selection. Gradients are hypothetical first/final NTP objectives. The old final-only shallow prediction is not a supported quality exit. Disposable100-update weights remain excluded from science initialization.')
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
data=ProbabilitySources(ROOT/p['data_relative']);x,y,mask=data.batch([1000000,1000001]);rows=[]
for label,path in [('initial',None),('final_only_u1530',old),('joint_disposable_u103',new)]:
 torch.manual_seed(p['seed']);m=probability_model(p)
 if path:
  s=torch.load(path,map_location='cpu',weights_only=False,mmap=True);m.load_state_dict(s['model']);del s
 grads=[];losses=[]
 for depth in (0,1):
  m.zero_grad(set_to_none=True);loss=reference_loss_sum(m,x,y,mask,depth)/mask.sum();loss.backward();losses.append(float(loss.detach()));grads.append({n:v.grad.detach().clone() for n,v in m.named_parameters()});del loss
 families={}
 for name,match in [('overall',lambda n:True)]+gradient_families(p):
  aa=bb=ab=0.
  for n,a in grads[0].items():
   if match(n):
    b=grads[1][n];aa+=float(a.double().square().sum());bb+=float(b.double().square().sum());ab+=float((a.double()*b.double()).sum())
  families[name]=dict(shallow_l2=aa**.5,deep_l2=bb**.5,cosine=ab/(aa*bb)**.5,joint_l2=max(0,.25*(aa+bb+2*ab))**.5)
 rows.append(dict(model=label,train_losses=losses,gradient_families=families));del m,grads
 if time.time()-start>600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>48*1024**3:raise RuntimeError('CPU diagnostic bound')
 result=dict(status='COMPLETE_TRAIN_GRADIENT_COMPATIBILITY',rows=rows,assigned_chips=0,optimizer_updates=0,heldout_examples=0,wall_seconds=time.time()-start,scope=plan['scope'])
 (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
