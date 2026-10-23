"""Independent FP32 CPU predictions on the whole fixed curve DEV panel."""
import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from compute_trainium.data import document_windows
from compute_trainium.independent import IndependentModel
from compute_trainium.metrics import DocumentScore,paired_group_interval,summarize
from compute_trainium.model import ModelConfig
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--original-root',type=Path,required=True);p.add_argument('--new-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--threads',type=int,default=16);args=p.parse_args();args.output.mkdir(exist_ok=False)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    started=time.monotonic();docs=sorted([d for d in json.loads(args.corpus.read_text())
        if d['split']=='dev' and d.get('cohort')=='curve'],key=lambda d:d['id'])
    spec={'corpus_sha256':file_sha256(args.corpus),'seeds':[23,47,71],'cohort':'curve','precision':'FP32 weights/matmuls, FP64 log normalization',
          'implementation':'Standalone separate-projection attention and real-pair RoPE, Torch CPU; no TorchTitan/native compiler',
          'partial_windows':'Trim only causal padding after every last scored target; preserve all actual prefix/target tokens',
          'scripts_sha256':file_sha256(__file__),'reference_sha256':file_sha256('compute_trainium/independent.py'),
          'started_unix':time.time(),'threads':args.threads}
    (args.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n')
    all_scores={};records=[]
    with torch.inference_mode():
        for arm in ['ordinary','recurrent']:
            for seed in spec['seeds']:
                run=(f'natural-curve-v1-ordinary-s{seed}-u3840' if arm=='ordinary' else f'preln-recurrent-v1-s{seed}-u3840')
                root=args.original_root if arm=='ordinary' else args.new_root
                checkpoint=root/'runs'/run/'checkpoint.pt';phase=root/'runs'/run/'report.json'
                while not phase.exists():
                    if time.time()>args.deadline_unix:raise TimeoutError('CPU verification deadline while waiting for frozen endpoint')
                    time.sleep(30)
                saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
                sig=saved['signature'];context=sig['context']
                if sig.get('dev_corpus_sha256')!=spec['corpus_sha256'] or saved['update']!=3840:raise ValueError('Endpoint/source identity differs')
                config=ModelConfig(**{k:v for k,v in sig['model'].items() if k in ModelConfig.__dataclass_fields__})
                model=IndependentModel(config,repetitions=1 if arm=='ordinary' else 2).eval();model.load_titan_state(saved['model'])
                checkpoint_sha=file_sha256(checkpoint);del saved
                scores=[];windows=targets=0;model_start=time.monotonic()
                for i,doc in enumerate(docs):
                    total=0.;count=0
                    for source,mask in document_windows(doc['tokens'],context):
                        n=int(mask.sum());hidden=model.hidden(source[None,:n])
                        for first in range(0,n,128):
                            end=min(n,first+128)
                            logits=F.linear(hidden[:,first:end],model.embedding.weight)[0].double()
                            target=source[first+1:end+1]
                            total+=float((logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1)).sum())
                        count+=n;windows+=1;targets+=n
                    if count!=len(doc['tokens']):raise AssertionError('CPU target coverage changed')
                    scores.append(DocumentScore(doc['domain'],doc['id'],doc['group'],total,count))
                    if (i+1)%100==0:print(json.dumps({'model':run,'documents':i+1,'seconds':time.monotonic()-model_start}),flush=True)
                    if time.time()>args.deadline_unix:raise TimeoutError('CPU verification deadline')
                (args.output/(run+'-scores.json')).write_text(json.dumps([asdict(s) for s in scores])+'\n')
                all_scores[arm,seed]=scores
                native_path=root/'runs'/(run+'-dev')/'report.json'
                while not native_path.exists():
                    if time.time()>args.deadline_unix:raise TimeoutError('Missing completed native reference')
                    time.sleep(2)
                native=json.loads(native_path.read_text());metrics=summarize(scores)
                records.append({'arm':arm,'seed':seed,'metrics':metrics,'checkpoint_sha256':checkpoint_sha,
                    'cpu_minus_native_nll':metrics['weighted_nll']-native['metrics']['weighted_nll'],
                    'windows':windows,'targets':targets,'seconds':time.monotonic()-model_start})
                (args.output/'progress.json').write_text(json.dumps({'records':records,'seconds':time.monotonic()-started},indent=2)+'\n')
                del model,scores;gc.collect()
    contrasts={str(s):paired_group_interval(all_scores['ordinary',s],all_scores['recurrent',s],draws=2000,seed=23) for s in spec['seeds']}
    report={'status':'COMPLETE_INDEPENDENT_CPU_CURVE_VERIFICATION','records':records,'contrasts':contrasts,
            'wall_seconds':time.monotonic()-started,'threads':args.threads,
            'scope':'Numerical verification on the same fixed DEV panel; does not replace native primary decisions or provide new model replications.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'contrasts':contrasts}),flush=True)


if __name__=='__main__':main()
