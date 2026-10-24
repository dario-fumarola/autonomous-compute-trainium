"""Independent FP32 DEV scoring, including cross-precision parent diagnostics.

Reports disagreement with an optional historical native score; it does not
turn that difference into a passed numerical-agreement qualification.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from compute_trainium.data import document_windows
from compute_trainium.independent import IndependentModel
from compute_trainium.metrics import DocumentScore, summarize
from compute_trainium.model import ModelConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--expected-checkpoint-sha256',required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--native-report',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--threads',type=int,default=12)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);started=time.time()
    if file_sha256(a.checkpoint)!=a.expected_checkpoint_sha256:raise AssertionError('Checkpoint changed')
    saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False);sig=saved['signature']
    if sig['dev_corpus_sha256']!=file_sha256(a.corpus):raise ValueError('Only the checkpoint\'s existing DEV may be scored')
    if saved['pauses'] or sig['model'].get('untied') or sig['model'].get('normalization')!='pre':
        raise ValueError('This scorer is confined to qualified pre-norm paths')
    if digest({'model':saved['model'],'optimizer':saved['optimizer']})!=saved['state_digest']:
        raise AssertionError('Stored complete state digest differs')
    c=ModelConfig(**{k:v for k,v in sig['model'].items() if k in ModelConfig.__dataclass_fields__})
    model=IndependentModel(c,repetitions=sig['model'].get('repetitions',1)).eval();model.load_titan_state(saved['model'])
    update,exposure=saved['update'],saved['real_tokens'];del saved
    docs=sorted([d for d in json.loads(a.corpus.read_text()) if d['split']=='dev' and d.get('cohort')=='curve'],key=lambda d:d['id'])
    if not docs:raise ValueError('Empty fixed DEV curve panel')
    scores=[]
    with torch.inference_mode():
        for i,d in enumerate(docs):
            total=0.;count=0
            for source,mask in document_windows(d['tokens'],sig['context']):
                n=int(mask.sum());hidden=model.hidden(source[None,:n])
                for first in range(0,n,128):
                    end=min(n,first+128);logits=F.linear(hidden[:,first:end],model.embedding.weight)[0].double()
                    target=source[first+1:end+1]
                    total+=float((logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1)).sum())
                count+=n
            if count!=len(d['tokens']):raise AssertionError('Lost real DEV targets')
            scores.append(DocumentScore(d['domain'],d['id'],d['group'],total,count))
            if (i+1)%150==0:print(json.dumps({'event':'cpu_documents','count':i+1,'seconds':time.time()-started}),flush=True)
            if time.time()>a.deadline_unix:raise TimeoutError('Independent CPU scoring deadline')
    metrics=summarize(scores);comparison=None
    if a.native_report:
        native=json.loads(a.native_report.read_text());delta=metrics['weighted_nll']-native['metrics']['weighted_nll']
        comparison={'cpu_minus_historical_native_nll':delta,'within_old_0_0002_agreement_limit':abs(delta)<=.0002,
                    'historical_native_training_precision':sig['precision'],
                    'scope':'Cross-precision diagnostic for a historical readout; disagreement is reported, never relabeled as a passed audit.'}
    if file_sha256(a.checkpoint)!=a.expected_checkpoint_sha256:raise AssertionError('Scoring changed source checkpoint')
    report={'status':'COMPLETE_INDEPENDENT_FP32_DEV_ANCHOR','update':update,'training_real_targets':exposure,
        'checkpoint_sha256':a.expected_checkpoint_sha256,'corpus_sha256':file_sha256(a.corpus),
        'scorer_sha256':file_sha256(__file__),'reference_sha256':file_sha256('compute_trainium/independent.py'),
        'metrics':metrics,'documents':len(scores),'targets':sum(s.tokens for s in scores),'comparison':comparison,
        'wall_seconds':time.time()-started,'native_chips_assigned':0,'test_model_scoring_performed':False,
        'cpu_seconds':{'user':resource.getrusage(resource.RUSAGE_SELF).ru_utime,'system':resource.getrusage(resource.RUSAGE_SELF).ru_stime},
        'scope':'Read-only existing DEV, independent FP32 attention/RoPE implementation and FP64 log normalization. Canonical parent score, not a numerical-agreement waiver, new training seed, research endpoint or TEST.'}
    write_state(a.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
