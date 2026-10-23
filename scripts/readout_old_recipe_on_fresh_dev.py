"""Exploratory old-recipe reference on the fresh DEV panel, never new TEST."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from compute_trainium.confirmation import paired_seed_interval
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
    specification={'started_unix':started,'corpus_sha256':file_sha256(a.corpus),'seeds':[23,47,71],
        'depths':{'ordinary':1,'shared':2},'source_sha256':file_sha256(__file__),
        'scope':'Exploratory reference motivated during LR screening. Read all six canonical old1B endpoints on fresh DEV, no new TEST, no checkpoint/exit selection or fit. Does not isolate beta2, since recipes also differ in warmup, tuning, topology, routing and initialization seeds.'}
    write_state(a.output/'specification.json',specification)
    for seed in specification['seeds']:
        for arm,depth in specification['depths'].items():
            run=f'natural-curve-v1-ordinary-s{seed}-u3840' if arm=='ordinary' else f'preln-recurrent-v1-s{seed}-u3840'
            root=a.root.parent if arm=='ordinary' else a.root
            checkpoint=root/'runs'/run/'checkpoint.pt';sha=file_sha256(checkpoint)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);sig=saved['signature']
            if saved['update']!=3840 or saved['pauses']!=0:raise ValueError('Not the canonical old endpoint')
            if arm=='shared' and sig['model'].get('normalization')!='pre':raise ValueError('Wrong normalization family')
            config=ModelConfig(**{k:v for k,v in sig['model'].items() if k in ModelConfig.__dataclass_fields__})
            model=IndependentModel(config,repetitions=depth).eval();model.load_titan_state(saved['model'])
            recipe={'optimizer_betas':list(saved['optimizer']['param_groups'][0]['betas']),
                'weight_decays':[g['weight_decay'] for g in saved['optimizer']['param_groups']],
                'schedule':saved['schedule'],'world':sig['world'],'batch':sig['batch'],'accumulation':sig['accumulation'],
                'training_real_targets':saved['real_tokens']};del saved
            begin=time.time();scores=[]
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
                    if count!=len(doc['tokens']):raise AssertionError('Fresh DEV target coverage differs')
                    scores.append(DocumentScore(doc['domain'],doc['id'],doc['group'],total,count))
                    if time.time()>a.deadline_unix:raise TimeoutError('Exploratory reference deadline')
            if file_sha256(checkpoint)!=sha:raise AssertionError('Read-only checkpoint changed')
            row={'arm':arm,'seed':seed,'depth':depth,'metrics':summarize(scores),'recipe':recipe,
                 'checkpoint_sha256':sha,'targets':sum(s.tokens for s in scores),'seconds':time.time()-begin}
            records.append(row);write_state(a.output/'progress.json',{'records':records});print(json.dumps(row),flush=True)
            del model,scores;gc.collect()
    lookup={(r['arm'],r['seed']):r['metrics']['weighted_nll'] for r in records}
    gain=paired_seed_interval({str(s):lookup['ordinary',s]-lookup['shared',s] for s in specification['seeds']})
    report={'status':'COMPLETE_OLD_RECIPE_FRESH_DEV_REFERENCE','specification':specification,'records':records,
            'paired_old_seed_gain':gain,'wall_seconds':time.time()-started,'configured_threads':12,
            'scope':'Independent FP32 numerical readout on fresh DEV. New recipe needs its own paired seed result; no across-recipe seed pairing or single-hyperparameter causal attribution.'}
    write_state(a.output/'report.json',report)
    write_state(Path('results/variable_depth_20261023/old-recipe-fresh-dev.json'),report)


if __name__=='__main__':main()
