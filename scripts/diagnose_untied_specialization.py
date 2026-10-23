"""Fixed-panel read-only projection of untied core parameter families.

A post-training projection is not a retrained shared solution or a new TEST
candidate. It can localize sensitivity, not identify the optimal sharing rule.
"""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.independent_untied import IndependentUntiedModel
from compute_trainium.training import file_sha256


def family(name):
 if name.startswith(('attn_norm','ffn_norm')):return 'norm'
 if name.startswith(('q.','k.','v.','o.')):return 'attention'
 return 'mlp'


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(16);torch.set_num_interop_threads(1);started=time.time()
 frozen=json.loads((a.root/'mechanism-states-v1/specification.json').read_text());selected=[d['id'] for d in frozen['documents'] if d['split']=='dev']
 data=a.root.parent/'natural-v2-1050m/dev.json';by_id={d['id']:d for d in json.loads(data.read_text())};docs=[by_id[k] for k in selected]
 spec={'identity':'untied-specialization-v1','frozen_unix':time.time(),'documents':selected,'targets_per_document':512,
  'modes':['baseline','project_norm','project_attention','project_mlp','project_all'],
  'source_sha256':file_sha256(data),'checkpoint':'untied-depth-v1-s23-u3840',
  'scope':'Exploratory single-checkpoint projection on pre-existing fixed natural DEV prefixes. Average each selected parameter pair and assign its midpoint to both copies. Original file unchanged; no training, model selection or TEST.'}
 (a.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n')
 cp=a.root/'runs/untied-depth-v1-s23-u3840/checkpoint.pt';saved=torch.load(cp,weights_only=False,mmap=True,map_location='cpu')
 config=ModelConfig(**{k:v for k,v in saved['signature']['model'].items() if k in ModelConfig.__dataclass_fields__});m=IndependentUntiedModel(config).eval();m.load_titan_state(saved['model']);del saved
 pairs=[];divergence=[]
 for j in range(m.core):
  first=dict(m.blocks[m.prelude+j].named_parameters());second=dict(m.extra[j].named_parameters())
  for name,p in first.items():
   q=second[name];x=p.detach().clone();y=q.detach().clone();f=family(name)
   pairs.append((p,q,x,y,f));aa=float(x.square().sum());bb=float(y.square().sum());delta=float((x-y).square().sum());dot=float((x*y).sum())
   divergence.append({'core_layer':j,'parameter':name,'family':f,'elements_per_copy':x.numel(),
    'first_norm':aa**.5,'second_norm':bb**.5,'cosine':dot/max((aa*bb)**.5,1e-30),'relative_pair_difference':delta**.5/max(((aa+bb)/2)**.5,1e-30)})
 rows=[]
 with torch.no_grad():
  for mode in spec['modes']:
   chosen=mode.removeprefix('project_')
   for p,q,x,y,f in pairs:
    if mode!='baseline' and chosen in [f,'all']:
     average=(x+y)*.5;p.copy_(average);q.copy_(average)
    else:p.copy_(x);q.copy_(y)
   per_doc=[]
   for d in docs:
    source=torch.tensor([1,*d['tokens'][:512]],dtype=torch.int64);h=m.hidden(source[:-1][None]);total=0.
    for first in range(0,512,128):
     end=first+128;logits=F.linear(h[:,first:end],m.embedding.weight)[0].double();target=source[first+1:end+1]
     total+=float((logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1)).sum())
    per_doc.append({'domain':d['domain'],'document':d['id'],'nll':total/512})
   domains={d:sum(x['nll'] for x in per_doc if x['domain']==d)/4 for d in ['web','math','code']}
   nll=sum(w*domains[d] for d,w in [('web',.6),('math',.2),('code',.2)])
   rows.append({'mode':mode,'weighted_nll':nll,'domains':domains,'documents':per_doc})
   print(json.dumps({'mode':mode,'weighted_nll':nll}),flush=True)
  for p,q,x,y,f in pairs:p.copy_(x);q.copy_(y)
 result={'status':'COMPLETE_UNTIED_COMPONENT_PROJECTION_DIAGNOSTIC','specification':spec,'rows':rows,'parameter_divergence':divergence,
  'elapsed_seconds':time.time()-started,'warning':'Projection changes weights without reoptimization, and averages can change parameter norms. Effects are not additive and do not prove which sharing restriction caused training loss. No projected weights are saved or eligible for TEST.'}
 (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
