"""Independent full-DEV FP32 checks of the predeclared within-model depth effect."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from compute_trainium.data import document_windows
from compute_trainium.independent import IndependentModel
from compute_trainium.metrics import DocumentScore,summarize
from compute_trainium.model import ModelConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(12);torch.set_num_interop_threads(1);started=time.time();records=[]
    docs=sorted([d for d in json.loads(a.corpus.read_text()) if d['split']=='dev' and d.get('cohort')=='curve'],key=lambda d:d['id'])
    for seed in [103,211,307]:
        run=f'vdepth-v1-mixture-s{seed}-u3840';checkpoint=a.root/'runs'/run/'checkpoint.pt'
        original_hash=file_sha256(checkpoint);saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        if saved['signature']['dev_corpus_sha256']!=file_sha256(a.corpus):raise ValueError('Source mismatch')
        config=ModelConfig(**{k:v for k,v in saved['signature']['model'].items() if k in ModelConfig.__dataclass_fields__})
        model=IndependentModel(config).eval();model.load_titan_state(saved['model']);del saved
        for depth in [1,3]:
            begin=time.time();model.repetitions=depth;scores=[]
            with torch.inference_mode():
                for doc in docs:
                    total=0.;count=0
                    for source,mask in document_windows(doc['tokens'],1024):
                        n=int(mask.sum());hidden=model.hidden(source[None,:n])
                        for first in range(0,n,128):
                            end=min(n,first+128);logits=F.linear(hidden[:,first:end],model.embedding.weight)[0].double()
                            target=source[first+1:end+1]
                            total+=float((logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1)).sum())
                        count+=n
                    if count!=len(doc['tokens']):raise AssertionError('CPU mechanism target coverage changed')
                    scores.append(DocumentScore(doc['domain'],doc['id'],doc['group'],total,count))
                    if time.time()>a.deadline_unix:raise TimeoutError('CPU mechanism audit deadline')
            native=json.loads((a.root/'runs'/f'{run}-dev-r{depth}'/'report.json').read_text())
            metrics=summarize(scores);difference=metrics['weighted_nll']-native['metrics']['weighted_nll']
            row={'seed':seed,'depth':depth,'metrics':metrics,'cpu_minus_native_nll':difference,
                 'targets':sum(s.tokens for s in scores),'seconds':time.time()-begin,'checkpoint_sha256':original_hash}
            records.append(row);write_state(a.output/'progress.json',{'records':records})
            if abs(difference)>.0002:raise AssertionError('Independent mechanism NLL differs from native')
            print(json.dumps({'seed':seed,'depth':depth,'difference':difference}),flush=True)
        if file_sha256(checkpoint)!=original_hash:raise AssertionError('Audit checkpoint changed')
        del model,scores;gc.collect()
    write_state(a.output/'report.json',{'status':'PASS_INDEPENDENT_DEPTH_MECHANISM_AUDIT','records':records,
        'wall_seconds':time.time()-started,'configured_threads':12,'scope':'Same full fresh DEV, independent FP32 model with FP64 log normalization; numerical audit, not new model replication. No TEST use.'})


if __name__=='__main__':main()
