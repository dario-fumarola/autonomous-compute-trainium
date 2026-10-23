"""Fixed natural-prefix train/DEV probe and per-application gradient readout."""
import argparse,copy,gc,json,random,sqlite3,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from compute_trainium.independent import IndependentModel
from compute_trainium.model import ModelConfig
from compute_trainium.diagnostics import traced_unroll,state_statistics,step_statistics
from compute_trainium.training import file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--old-root',type=Path,required=True);p.add_argument('--new-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(exist_ok=False);torch.set_num_threads(16);torch.set_num_interop_threads(1);started=time.time()
 db=sqlite3.connect('file:'+str(a.data/'documents.sqlite')+'?mode=ro',uri=True);docs=[]
 for domain in ['web','math','code']:
  train=db.execute("select id,offset,length from documents where domain=? and split='train' order by id",(domain,)).fetchall()
  random.Random(f'23:{domain}').shuffle(train)
  # Freeze early actual training-order documents, guaranteed seen before 100M.
  chosen=[r for r in train[:100] if r[2]>=512][:4]
  dev=db.execute("select id,offset,length from documents where domain=? and split='dev' and cohort='curve' and length>=512 order by id limit 4",(domain,)).fetchall()
  array=np.memmap(a.data/(domain+'.u16'),mode='r',dtype=np.uint16)
  for split,selected in [('train',chosen),('dev',dev)]:
   if len(selected)!=4:raise ValueError('Fixed natural probe unavailable')
   for identity,offset,length in selected:
    tokens=torch.tensor(np.array(array[offset:offset+512],dtype=np.int64));sequence=torch.cat([torch.tensor([1]),tokens])
    docs.append({'id':identity,'domain':domain,'split':split,'sequence':sequence})
 spec={'identity':'recurrent-states-v1','frozen_unix':time.time(),'documents':[{k:v for k,v in d.items() if k!='sequence'} for d in docs],
       'source_sha256':file_sha256(a.data/'source.json'),'targets_per_document':512,'precision':'FP32 CPU independent reference',
       'selection':'Four length>=512 documents/domain from first100 actual domain training-order documents; first four ID-sorted qualifying curve DEV documents/domain. First512 targets; BOS; isolated document format in both splits.',
       'gradient_selection':'First frozen TRAIN document/domain; seed23 terminal recurrent model; split each repeated core use into equal-weight copies only for gradient measurement',
       'scope':'Descriptive fixed-prefix diagnosis, not population TRAIN NLL or a new primary outcome; no TEST, model update or LR selection'}
 (a.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n');records=[]
 for arm in ['ordinary','recurrent']:
  for seed in [23,47,71]:
   cp=(a.old_root/f'runs/natural-curve-v1-ordinary-s{seed}-u3840/checkpoint.pt' if arm=='ordinary' else a.new_root/f'runs/preln-recurrent-v1-s{seed}-u3840/checkpoint.pt')
   saved=torch.load(cp,weights_only=False,mmap=True,map_location='cpu');cfg=ModelConfig(**{k:v for k,v in saved['signature']['model'].items() if k in ModelConfig.__dataclass_fields__})
   m=IndependentModel(cfg,repetitions=1 if arm=='ordinary' else 2).eval();m.load_titan_state(saved['model']);del saved
   with torch.no_grad():
    for d in docs:
     x=d['sequence'][:-1][None];target=d['sequence'][1:];pair={}
     for r in [1,2]:
      hidden,e,states=traced_unroll(m,x,r)
      logits=F.linear(hidden,m.embedding.weight)[0].double();loss=logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1)
      pair[str(r)]={'nll':float(loss.mean()),'prelude':state_statistics(e),'passes':[state_statistics(s) for s in states]}
      if r==2:pair[str(r)]['between_passes']=step_statistics(*states)
     records.append({'arm':arm,'seed':seed,'document':d['id'],'domain':d['domain'],'split':d['split'],'readouts':pair})
   (a.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'event':'state_probe','arm':arm,'seed':seed,'records':len(records)}),flush=True)
   if arm=='recurrent' and seed==23:
    extra=torch.nn.ModuleList([copy.deepcopy(m.blocks[i]) for i in range(m.prelude,m.prelude+m.core)])
    first=[p for i in range(m.prelude,m.prelude+m.core) for p in m.blocks[i].parameters()];second=list(extra.parameters())
    accum1=[torch.zeros_like(p) for p in first];accum2=[torch.zeros_like(p) for p in second];gradient_rows=[]
    for domain in ['web','math','code']:
     d=next(d for d in docs if d['split']=='train' and d['domain']==domain);x=d['sequence'][:-1][None];target=d['sequence'][1:]
     m.zero_grad(set_to_none=True);extra.zero_grad(set_to_none=True)
     h,_,_=traced_unroll(m,x,2,extra);logits=F.linear(h,m.embedding.weight)[0]
     loss=F.cross_entropy(logits,target);loss.backward()
     dot=aa=bb=0.
     for p,q,g1,g2 in zip(first,second,accum1,accum2,strict=True):
      g1.add_(p.grad/3);g2.add_(q.grad/3);dot+=float((p.grad*q.grad).sum());aa+=float(p.grad.square().sum());bb+=float(q.grad.square().sum())
     gradient_rows.append({'domain':domain,'loss':float(loss.detach()),'first_use_norm':aa**.5,'second_use_norm':bb**.5,'cosine':dot/max((aa*bb)**.5,1e-30),
       'cancellation_ratio':max(0,aa+bb+2*dot)**.5/max(aa**.5+bb**.5,1e-30)})
    dot=sum(float((x*y).sum()) for x,y in zip(accum1,accum2));aa=sum(float(x.square().sum()) for x in accum1);bb=sum(float(x.square().sum()) for x in accum2)
    gradient={'rows':gradient_rows,'mean_gradient_cosine':dot/max((aa*bb)**.5,1e-30),'first_use_mean_gradient_norm':aa**.5,'second_use_mean_gradient_norm':bb**.5,
      'mean_gradient_cancellation_ratio':max(0,aa+bb+2*dot)**.5/max(aa**.5+bb**.5,1e-30),'scope':'Three fixed natural TRAIN prefixes; FP32 partial derivatives by application at unchanged seed23 checkpoint; no optimizer update. Correlation is not a causal test of sharing.'}
    (a.output/'application-gradients.json').write_text(json.dumps(gradient,indent=2)+'\n');del extra,accum1,accum2,first,second
   del m;gc.collect()
 (a.output/'report.json').write_text(json.dumps({'status':'COMPLETE_RECURRENT_STATE_DIAGNOSTICS','specification':spec,'records':records,'wall_seconds':time.time()-started},indent=2)+'\n')


if __name__=='__main__':main()
