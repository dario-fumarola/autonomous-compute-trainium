"""Read-only precision calibration on the exact natural TRAIN probe batch."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from compute_trainium.depth_mixture import routed_indices
from compute_trainium.independent import IndependentModel
from compute_trainium.metrics import real_token_losses
from compute_trainium.model import ModelConfig
from compute_trainium.training import PackedCorpus,file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--packed',type=Path,required=True)
    p.add_argument('--depth',type=int,required=True);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(12);torch.set_num_interop_threads(1);started=time.time()
    saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False);sig=saved['signature']
    if sig['corpus_sha256']!=file_sha256(a.corpus):raise ValueError('Source mismatch')
    data=PackedCorpus.from_mapped(a.packed,sig['context'],sig['data_seed'],sig['corpus_sha256'])
    ids=routed_indices(saved['cursor'],0,a.rank,sig['world'],sig['batch'],sig['accumulation'],sig['routing_seed'])
    batch=data.batch(ids)[0]
    config=ModelConfig(**{k:v for k,v in sig['model'].items() if k in ModelConfig.__dataclass_fields__})
    results={};reference=None
    for bf16 in [False,True]:
        model=IndependentModel(config,repetitions=a.depth).eval();model.load_titan_state(saved['model'])
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):logits=model(batch.inputs)
        loss=real_token_losses(logits,batch.targets,batch.target_mask).sum()/batch.target_mask.sum();loss.backward()
        values=[v.grad.detach().clone() for v in model.parameters()]
        results['bf16' if bf16 else 'fp32']={'loss':float(loss.detach()),'gradient_norm':sum(float(v.square().sum()) for v in values)**.5}
        if not bf16:reference=values
        else:
            difference=sum(float((v-w).square().sum()) for v,w in zip(values,reference,strict=True))
            total=sum(float(w.square().sum()) for w in reference)
            results['bf16_to_fp32_gradient_relative_l2']=(difference/total)**.5
        del model,logits,loss;gc.collect()
    report={'status':'COMPLETE_CPU_PRECISION_CALIBRATION','checkpoint_sha256':file_sha256(a.checkpoint),
       'rank':a.rank,'depth':a.depth,'source_cursor':saved['cursor'],'real_targets':int(batch.target_mask.sum()),
       'results':results,'seconds':time.time()-started,
       'scope':'Same independent model, weights and natural TRAIN windows, CPU FP32 versus CPU BF16 only. No native gradient vectors are available here; this does not directly identify native-to-FP32 error or the cause of cross-backend differences. No optimizer step or TEST use.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
