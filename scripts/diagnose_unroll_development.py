"""Fixed-prefix same-weight pass dependence across preserved training ages."""
import argparse,gc,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.independent import IndependentModel
from compute_trainium.diagnostics import traced_unroll


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(16);started=time.time()
 fixed=json.loads((a.root/'mechanism-states-v1/specification.json').read_text());ids=[d['id'] for d in fixed['documents'] if d['split']=='dev']
 docs={d['id']:d for d in json.loads((a.root.parent/'natural-v2-1050m/dev.json').read_text())}
 spec={'identity':'unroll-development-v1','seed':23,'updates':[384,1152,3456,3840],'documents':ids,'targets_per_document':512,
  'passes':[1,2],'frozen_unix':time.time(),'scope':'Exploratory read-only history on the pre-existing fixed DEV prefix panel; one-pass readout is outside training configuration. Not the primary curve or TEST. Last point is annealed; earlier ones are stable.'}
 (a.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n');rows=[]
 with torch.no_grad():
  for update in spec['updates']:
   v=torch.load(a.root/f'runs/preln-recurrent-v1-s23-u{update}/checkpoint.pt',weights_only=False,mmap=True,map_location='cpu')
   cfg=ModelConfig(**{k:x for k,x in v['signature']['model'].items() if k in ModelConfig.__dataclass_fields__});model=IndependentModel(cfg,repetitions=2).eval();model.load_titan_state(v['model']);exposure=v['real_tokens'];del v
   for passes in spec['passes']:
    per_doc=[]
    for identity in ids:
     d=docs[identity];source=torch.tensor([1,*d['tokens'][:512]],dtype=torch.int64);hidden=traced_unroll(model,source[:-1][None],passes)[0];total=0.
     for first in range(0,512,128):
      end=first+128;logits=F.linear(hidden[:,first:end],model.embedding.weight)[0].double();target=source[first+1:end+1]
      total+=float((logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1)).sum())
     per_doc.append({'domain':d['domain'],'document':identity,'nll':total/512})
    domains={d:sum(x['nll'] for x in per_doc if x['domain']==d)/4 for d in ['web','math','code']};nll=sum(w*domains[d] for d,w in [('web',.6),('math',.2),('code',.2)])
    rows.append({'update':update,'real_targets':exposure,'passes':passes,'nll':nll,'domains':domains})
   print(json.dumps({'update':update,'one_pass':rows[-2]['nll'],'two_pass':rows[-1]['nll'],'gain':rows[-2]['nll']-rows[-1]['nll']}),flush=True)
   del model;gc.collect()
 (a.output/'report.json').write_text(json.dumps({'status':'COMPLETE_FIXED_PREFIX_UNROLL_DEVELOPMENT','specification':spec,'rows':rows,'elapsed_seconds':time.time()-started},indent=2)+'\n')


if __name__=='__main__':main()
