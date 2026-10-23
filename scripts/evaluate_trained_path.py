"""Read-only evaluation of the exact trained path, with explicit frozen TEST gate."""
import argparse
import copy
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

cores=os.environ['RESEARCH_VISIBLE_CORES'].split(',')
os.environ['NEURON_RT_VISIBLE_CORES']=cores[int(os.environ.get('LOCAL_RANK','0'))]

import torch
import torch.distributed as dist

from compute_trainium.data import document_windows, layout
from compute_trainium.metrics import DocumentScore, summarize
from compute_trainium.model import ModelConfig
from compute_trainium.recurrence import config_from_signature, build_from_config
from compute_trainium.training import TokenObjective, file_sha256
from compute_trainium.evaluation_gate import validate_trained_test_release
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repetitions',type=int,choices=[1,2,4],required=True)
    p.add_argument('--cohort',choices=['curve','tune','all'],default='curve')
    p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--test-release',type=Path)
    p.add_argument('--calibration-report',type=Path)
    args=p.parse_args();started=time.monotonic();torch.set_num_threads(2);torch.neuron.set_device(0)
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    dist.init_process_group('neuron')
    args.output.mkdir(parents=True,exist_ok=True)
    corpus_sha=file_sha256(args.corpus)
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    sig=saved['signature'];context=sig['context'];exposure=saved['real_tokens']
    if sig['precision']!='bf16' or saved['pauses']!=0:raise ValueError('This readout is for the fixed pre-norm ordinary/recurrent paths')
    if 'repetitions' in sig['model'] and sig['model'].get('normalization')!='pre':
        raise ValueError('The failed sandwich screen is not part of this readout')
    if args.split=='test':
        if args.test_release is None:raise ValueError('Sealed TEST requires its frozen release manifest')
        release=json.loads(args.test_release.read_text())
        checkpoint_sha=file_sha256(args.checkpoint)
        validate_trained_test_release(release,corpus_sha=corpus_sha,checkpoint_sha=checkpoint_sha,
            repetitions=args.repetitions,evaluator_sha=file_sha256(__file__),trained_repetitions=sig['model'].get('repetitions',1))
    elif corpus_sha!=sig.get('dev_corpus_sha256',sig['corpus_sha256']):
        raise ValueError('DEV source differs from checkpoint')
    trained=sig['model'].get('repetitions',1)
    if args.repetitions!=trained:raise ValueError('This evaluator permits the trained path only')
    config=config_from_signature(sig)
    torch.manual_seed(0);model=build_from_config(config)
    model.load_state_dict(saved['model'],strict=True);before=digest(model.state_dict());del saved;gc.collect()
    reference=copy.deepcopy(model).eval() if rank==0 else None
    model.to('neuron').eval()
    corpus=json.loads(args.corpus.read_text())
    docs=sorted([d for d in corpus if d['split']==args.split and
                 (args.split=='test' or args.cohort=='all' or d.get('cohort')==args.cohort)],key=lambda d:d['id'])
    if not docs:raise ValueError('Empty frozen cohort')
    # Every worker has identical shapes; uneven final document counts are fine.
    first_source,first_mask=next(document_windows(docs[0]['tokens'],context))
    first=layout(first_source[None],first_mask[None])
    objective=TokenObjective(model,first.prediction_positions,True)
    compiled=torch.compile(objective,backend='neuron',fullgraph=True,dynamic=False)
    cpu_check=None;scores=[];windows=0
    with torch.no_grad():
        probe=compiled(*(v.contiguous().to('neuron') for v in [first.inputs,first.targets,first.target_mask]))
        if rank==0:
            torch.set_num_threads(8)
            ref=TokenObjective(reference,first.prediction_positions,True,'cpu')
            cpu=float(ref(first.inputs,first.targets,first.target_mask));native=float(probe.detach().cpu())
            n=int(first.target_mask.sum())
            if abs(native-cpu)/n>.03:raise AssertionError('Distributed readout probe differs from CPU')
            cpu_check={'native_nll':native/n,'cpu_nll':cpu/n}
            del reference,ref;torch.set_num_threads(2)
        dist.barrier()
        for doc in docs[rank::world]:
            total=0.;count=0
            for source,mask in document_windows(doc['tokens'],context):
                b=layout(source[None],mask[None])
                value=float(compiled(*(v.contiguous().to('neuron') for v in [b.inputs,b.targets,b.target_mask])).detach().cpu())
                if not torch.isfinite(torch.tensor(value)):raise AssertionError('Nonfinite held-out NLL')
                total+=value;count+=int(mask.sum());windows+=1
            if count!=len(doc['tokens']):raise AssertionError('Lost real held-out targets')
            scores.append(DocumentScore(doc['domain'],doc['id'],doc['group'],total,count))
    if digest(model.state_dict())!=before:raise AssertionError('Evaluation changed full model weights')
    (args.output/f'rank-{rank}.json').write_text(json.dumps({'scores':[asdict(s) for s in scores],
        'windows':windows,'weights_unchanged':True,'state_digest':before})+'\n')
    dist.barrier()
    if rank==0:
        parts=[json.loads((args.output/f'rank-{r}.json').read_text()) for r in range(world)]
        if len({p['state_digest'] for p in parts})!=1:raise AssertionError('Replica weights differ')
        scores=[DocumentScore(**s) for part in parts for s in part['scores']]
        if {s.document for s in scores}!={d['id'] for d in docs} or len(scores)!=len(docs):
            raise AssertionError('Distributed document coverage/pairing differs')
        metrics=summarize(scores)
        calibration=None
        if args.calibration_report:
            expected=json.loads(args.calibration_report.read_text())
            calibration=metrics['weighted_nll']-expected['metrics']['weighted_nll']
            if abs(calibration)>.0001:raise AssertionError('Sharded readout differs from original scalar readout')
        report={'status':'PASS_SHARDED_TRAINED_PATH_READOUT','split':args.split,'cohort':args.cohort,
                'training_real_tokens':exposure,'training_repetitions':sig['model'].get('repetitions',1),
                'inference_repetitions':args.repetitions,'parameters':config.parameter_count(),
                'world':world,'documents':len(scores),'targets':sum(s.tokens for s in scores),
                'scored_windows':sum(p['windows'] for p in parts),'native_probe_windows':world,
                'metrics':metrics,'cpu_probe':cpu_check,'weights_unchanged_all_replicas':True,
                'scalar_calibration_delta_nats':calibration,'corpus_sha256':corpus_sha,
                'forward_flops_per_window':config.flops(context,context)['forward_matmul_flops'],
                'wall_seconds':time.monotonic()-started,'runtime_config':asdict(config),
                'scope':'Exact frozen trained path. No optimizer update, inference-depth intervention, or model selection.'}
        (args.output/'document-scores.json').write_text(json.dumps([asdict(s) for s in scores])+'\n')
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
