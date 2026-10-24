"""Exploratory completed-decay short-exposure reference, DEV only.

Selection remains the already frozen TUNE choice. These single-seed references
do not replace the current primary analysis or establish a scaling law.
"""
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
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);started=time.time();records=[]
    torch.set_num_threads(12);torch.set_num_interop_threads(1)
    selection=a.campaign/'tuning.json';selection_sha=file_sha256(selection);corpus_sha=file_sha256(a.corpus)
    chosen=json.loads(selection.read_text())['selected']
    docs=sorted([d for d in json.loads(a.corpus.read_text()) if d['split']=='dev' and d.get('cohort')=='curve'],key=lambda d:d['id'])
    if not docs:raise ValueError('Missing DEV curve cohort')
    specification={'identity':'selected-tune-decay-reference-v1','started_unix':started,'deadline_unix':a.deadline_unix,
        'selected_by':'Previously frozen TUNE only','selection_sha256':selection_sha,'corpus_sha256':corpus_sha,
        'models':{arm:chosen[arm]['run'] for arm in ['ordinary','fixed','mixture']},
        'source_sha256':{f:file_sha256(f) for f in ['scripts/readout_selected_tunes.py','compute_trainium/independent.py','compute_trainium/model.py','compute_trainium/data.py','compute_trainium/metrics.py']},
        'configured_threads':12,'scope':'Read all three previously selected seed11 completed-decay 301M models on common DEV. No new fit, LR/exit/model selection, TEST scoring or primary-analysis change. Different seeds and schedules prevent treating these and the main curves as one uninterrupted trajectory. Three exposure points cannot validate a long-horizon scaling law.'}
    write_state(a.output/'specification.json',specification)
    try:
        initial=set();exposures=set();source=set()
        for arm in ['ordinary','fixed','mixture']:
            run=chosen[arm]['run'];checkpoint=a.root/'runs'/run/'checkpoint.pt';sha=file_sha256(checkpoint)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);signature=saved['signature']
            if saved['update']!=1152 or signature['seed']!=11 or saved['pauses'] or signature['dev_corpus_sha256']!=corpus_sha:
                raise ValueError('Not the declared completed-decay tuning checkpoint')
            if saved['real_tokens']!=saved['schedule']['decay_end'] or signature['model'].get('normalization')!='pre':
                raise ValueError('Missing completed annealing or wrong model family')
            initial.add(saved['initial_model_digest']);exposures.add(saved['real_tokens']);source.add(signature['corpus_sha256'])
            if len(initial)!=1 or len(exposures)!=1 or len(source)!=1:raise ValueError('Tuning arms are not paired')
            config=ModelConfig(**{k:v for k,v in signature['model'].items() if k in ModelConfig.__dataclass_fields__})
            depth=1 if arm=='ordinary' else 2
            model=IndependentModel(config,repetitions=depth).eval();model.load_titan_state(saved['model'])
            if sum(p.numel() for p in model.parameters())!=150436608:raise ValueError('Parameter count differs')
            exposure=saved['real_tokens'];schedule=saved['schedule'];del saved
            begin=time.time();scores=[]
            with torch.inference_mode():
                for index,doc in enumerate(docs):
                    total=0.;count=0
                    for tokens,mask in document_windows(doc['tokens'],1024):
                        n=int(mask.sum());hidden=model.hidden(tokens[None,:n])
                        for first in range(0,n,128):
                            end=min(n,first+128);logits=F.linear(hidden[:,first:end],model.embedding.weight)[0].double()
                            target=tokens[first+1:end+1]
                            total+=float((logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1)).sum())
                        count+=n
                    if count!=len(doc['tokens']):raise AssertionError('DEV target coverage differs')
                    scores.append(DocumentScore(doc['domain'],doc['id'],doc['group'],total,count))
                    if (index+1)%150==0:print(json.dumps({'arm':arm,'documents':index+1}),flush=True)
                    if time.time()>a.deadline_unix:raise TimeoutError('Exploratory reference deadline')
            if file_sha256(checkpoint)!=sha or file_sha256(selection)!=selection_sha or file_sha256(a.corpus)!=corpus_sha:
                raise AssertionError('Read-only inputs changed')
            row={'arm':arm,'identity':run,'seed':11,'depth':depth,'training_real_targets':exposure,
                'schedule':schedule,'checkpoint_sha256':sha,'metrics':summarize(scores),
                'targets':sum(s.tokens for s in scores),'active_seconds':time.time()-begin}
            records.append(row);write_state(a.output/'progress.json',{'records':records})
            print(json.dumps({'arm':arm,'nll':row['metrics']['weighted_nll']}),flush=True)
            del model,scores;gc.collect()
        report={'status':'COMPLETE_SELECTED_TUNE_DECAY_DEV_REFERENCE','specification':specification,'records':records,
            'wall_seconds':time.time()-started,'configured_cpu_threads':12,'native_chips_assigned':0,
            'scope':specification['scope']+' CPU work overlaps the live campaign; elapsed times are not added to node time.'}
        write_state(a.output/'report.json',report)
    except Exception as error:
        write_state(a.output/'report.json',{'status':'STOPPED_SELECTED_TUNE_REFERENCE','error_type':type(error).__name__,
            'records':records,'wall_seconds':time.time()-started});raise


if __name__=='__main__':main()
