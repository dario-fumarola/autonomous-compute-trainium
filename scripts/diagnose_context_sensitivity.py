"""Natural-prefix output diversity; diagnostic only, no model fitting."""
import argparse
import json
import time
from pathlib import Path

import torch

from compute_trainium.data import document_windows
from compute_trainium.recurrence import build_from_config, config_from_signature


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(8);args.output.mkdir(parents=True,exist_ok=False)
    spec={'selection':'First eight ID-sorted tuning DEV documents/domain; positions 64:128 of first original window',
          'models':['ordinary_pre_norm','single_sandwich','two_pass_sandwich'],'purpose':'Context sensitivity diagnostic, not a new outcome criterion',
          'created_unix':time.time()}
    (args.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n')
    corpus=json.loads(args.corpus.read_text())
    docs=[d for domain in ['web','math','code'] for d in sorted([d for d in corpus if d['domain']==domain and d['split']=='dev' and d['cohort']=='tune'],key=lambda d:d['id'])[:8]]
    paths=[Path('/work/runs/natural-curve-v1-lr-ordinary-0/checkpoint.pt'),
           Path('/work/recurrence/runs/recurrent-natural-v1-lr-single-0/checkpoint.pt'),
           Path('/work/recurrence/runs/recurrent-natural-v1-lr-recurrent-0/checkpoint.pt')]
    result={}
    with torch.inference_mode():
        for name,path in zip(spec['models'],paths):
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            model=build_from_config(config_from_signature(saved['signature'])).eval();model.load_state_dict(saved['model']);del saved
            pieces=[]
            for doc in docs:
                source,mask=next(document_windows(doc['tokens'],1024))
                with torch.autocast('cpu',dtype=torch.bfloat16):logits=model(source[None,:-1],prediction_positions=torch.arange(64,128))
                pieces.append(logits[0][mask[64:128]].float())
            logits=torch.cat(pieces);logp=logits.log_softmax(-1);prob=logp.exp();average=prob.mean(0)
            information=(prob*(logp-average.log())).sum(-1).mean()
            result[name]={'natural_positions':len(logits),'mean_KL_to_average_distribution':float(information),
                          'mean_output_entropy':float(-(prob*logp).sum(-1).mean()),
                          'mean_centered_logit_std_across_positions':float((logits-logits.mean(-1,keepdim=True)).std(0).mean())}
            print(json.dumps({name:result[name]}),flush=True)
            del model
    (args.output/'report.json').write_text(json.dumps({'status':'COMPLETE_NATURAL_CONTEXT_SENSITIVITY','results':result,'specification':spec},indent=2)+'\n')


if __name__=='__main__':main()
