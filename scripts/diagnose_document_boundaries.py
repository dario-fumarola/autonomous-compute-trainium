"""Same tokens/positions/targets, with/without cross-document attention."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from compute_trainium.independent import IndependentModel
from compute_trainium.model import ModelConfig
from compute_trainium.training import file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--old-root',type=Path,required=True);p.add_argument('--new-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(exist_ok=False);torch.set_num_threads(16);torch.set_num_interop_threads(1);started=time.time()
 data=a.old_root/'natural-v2-1050m';tokens=np.load(data/'packed/tokens.npy',mmap_mode='r');domains=np.load(data/'packed/domains.npy',mmap_mode='r')
 windows=[]
 for i in sorted(np.random.default_rng(20261023).choice(3456*256,256,replace=False).tolist()):
  begin=i*1024;x=np.asarray(tokens[begin:begin+1024]);d=np.asarray(domains[begin:begin+1024]);boundary=(d==0)&(x==1)
  if boundary[1:].any():windows.append(i)
  if len(windows)==32:break
 if len(windows)!=32:raise ValueError('Frozen sample is unavailable')
 spec={'window_indices':windows,'seed':23,'source_sha256':file_sha256(data/'source.json'),
  'protocol_sha256':file_sha256('docs/DOCUMENT_BOUNDARY_DIAGNOSTIC.md'),'context':1024,
  'frozen_unix':time.time(),'precision':'FP32 CPU independent model, FP64 log normalization',
  'scope':'Read-only TRAIN context intervention; original tokens, positions, targets and loss masks retained; no synthetic corpus'}
 (a.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n');records=[]
 for arm in ['ordinary','recurrent']:
  path=(a.old_root/'runs/natural-curve-v1-ordinary-s23-u3840/checkpoint.pt' if arm=='ordinary' else a.new_root/'runs/preln-recurrent-v1-s23-u3840/checkpoint.pt')
  saved=torch.load(path,weights_only=False,mmap=True,map_location='cpu');config=ModelConfig(**{k:v for k,v in saved['signature']['model'].items() if k in ModelConfig.__dataclass_fields__})
  m=IndependentModel(config,repetitions=1 if arm=='ordinary' else 2).eval();m.load_titan_state(saved['model']);del saved
  with torch.no_grad():
   for index in windows:
    begin=index*1024;source=torch.tensor(np.asarray(tokens[begin:begin+1025]).copy(),dtype=torch.int64);labels=torch.tensor(np.asarray(domains[begin+1:begin+1025]).copy())
    segment=torch.tensor(np.asarray((domains[begin:begin+1024]==0)&(tokens[begin:begin+1024]==1)).cumsum(),dtype=torch.int64)
    same=segment[:,None]==segment[None,:];mask=same&torch.ones(1024,1024,dtype=torch.bool).tril()
    for mode in ['packed','document_masked']:
     h=m.hidden(source[:-1][None],attention_mask=mask if mode=='document_masked' else None);losses=[]
     for first in range(0,1024,128):
      end=first+128;logits=F.linear(h[:,first:end],m.embedding.weight)[0].double();target=source[first+1:end+1]
      losses.append(logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1))
     loss=torch.cat(losses)
     records.append({'arm':arm,'mode':mode,'window':index,'domains':{d:{'tokens':int((labels==j).sum()),'nll_sum':float(loss[labels==j].sum())} for d,j in [('web',1),('math',2),('code',3)]}})
  del m;gc.collect();print(json.dumps({'event':'boundary_probe','arm':arm,'records':len(records)}),flush=True)
 aggregate=[]
 for arm in ['ordinary','recurrent']:
  for mode in ['packed','document_masked']:
   rows=[r for r in records if r['arm']==arm and r['mode']==mode]
   stats={d:sum(r['domains'][d]['nll_sum'] for r in rows)/sum(r['domains'][d]['tokens'] for r in rows) for d in ['web','math','code']}
   aggregate.append({'arm':arm,'mode':mode,'nll':sum(w*stats[d] for d,w in [('web',.6),('math',.2),('code',.2)]),'domains':stats})
 result={'status':'COMPLETE_FIXED_DOCUMENT_BOUNDARY_DIAGNOSTIC','specification':spec,'aggregate':aggregate,'records':records,'elapsed_seconds':time.time()-started}
 (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'aggregate':aggregate}),flush=True)


if __name__=='__main__':main()
