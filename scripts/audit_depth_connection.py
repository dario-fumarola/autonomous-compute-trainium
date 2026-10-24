"""Versioned fixed-depth/connection control audit, based on audit_variable_depth.py.

Same independent scoring/copy checks, with explicit family and DEV-cohort routing.
"""
import argparse
import gc
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from compute_trainium.data import document_windows
from compute_trainium.independent import IndependentModel
from compute_trainium.reinjection_control import IndependentReinjectionControl
from compute_trainium.metrics import DocumentScore,summarize
from compute_trainium.model import ModelConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--cohort',choices=['curve','tune'],required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True);torch.set_num_threads(12);torch.set_num_interop_threads(1)
    started=time.time();previous=json.loads((a.output/'progress.json').read_text())['records'] if (a.output/'progress.json').exists() else [];docs=sorted([d for d in json.loads(a.corpus.read_text()) if d['split']=='dev' and d['cohort']==a.cohort],key=lambda d:d['id'])
    corpus_sha=file_sha256(a.corpus);records=[]
    pending=json.loads(a.models.read_text())
    while pending:
        model=next((m for m in pending if (a.root/'runs'/m['evaluation_identity']/'report.json').exists()),None)
        if model is None:
            if time.time()>a.deadline_unix:raise TimeoutError('Endpoint audit waiting deadline')
            time.sleep(5);continue
        pending.remove(model)
        existing=next((r for r in previous if r['identity']==model['identity']),None)
        if existing:
            copy=a.output/'recovery'/model['identity']/'checkpoint.pt'
            original=a.root/'runs'/model['identity']/'checkpoint.pt'
            if file_sha256(copy)!=existing['checkpoint_sha256'] or file_sha256(original)!=existing['checkpoint_sha256']:
                raise AssertionError('Resumed audit endpoint/copy changed')
            records.append(existing);continue
        depth=2 if model['arm']=='fixed' else 1
        run=model['identity'];native=a.root/'runs'/model['evaluation_identity']/'report.json'
        while not native.exists():
            if time.time()>a.deadline_unix:raise TimeoutError('Endpoint audit waiting deadline')
            time.sleep(5)
        begin=time.time();checkpoint=a.root/'runs'/run/'checkpoint.pt'
        saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False);sig=saved['signature']
        if sig['dev_corpus_sha256']!=corpus_sha:raise ValueError('Audit source changed')
        recovered=digest({'model':saved['model'],'optimizer':saved['optimizer']})
        if recovered!=saved['state_digest']:raise AssertionError('Serialized full model/optimizer state differs')
        if {tuple(g['betas']) for g in saved['optimizer']['param_groups']}!={(.9,.95)}:
            raise AssertionError('Unexpected optimizer recipe')
        if sorted(g['weight_decay'] for g in saved['optimizer']['param_groups'])!=[0.,.1]:
            raise AssertionError('Unexpected decay groups')
        c=ModelConfig(**{k:v for k,v in sig['model'].items() if k in ModelConfig.__dataclass_fields__})
        if sig['precision']!='fp32' or sig['model']['repetitions']!=depth:
            raise ValueError('Declared fixed/one-pass FP32 control path required')
        cheap=bool(sig['model'].get('reinjection_only'))
        if cheap!=(model['arm']=='reinjection'):
            raise ValueError('Inventory and model-family marker disagree')
        cls=IndependentReinjectionControl if cheap else IndependentModel
        cpu=cls(c,repetitions=depth).eval();cpu.load_titan_state(saved['model'])
        exposure=saved['real_tokens'];del saved;checkpoint_sha=file_sha256(checkpoint)
        scores=[]
        with torch.inference_mode():
            for index,d in enumerate(docs):
                total=0.;count=0
                for source,mask in document_windows(d['tokens'],1024):
                    n=int(mask.sum());hidden=cpu.hidden(source[None,:n])
                    for first in range(0,n,128):
                        end=min(n,first+128);logits=F.linear(hidden[:,first:end],cpu.embedding.weight)[0].double()
                        target=source[first+1:end+1]
                        total+=float((logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1)).sum())
                    count+=n
                if count!=len(d['tokens']):raise AssertionError('Lost CPU-scored real target')
                scores.append(DocumentScore(d['domain'],d['id'],d['group'],total,count))
                if (index+1)%150==0:print(json.dumps({'run':run,'cpu_documents':index+1}),flush=True)
                if time.time()>a.deadline_unix:raise TimeoutError('Endpoint audit scoring deadline')
        metrics=summarize(scores);native_report=json.loads(native.read_text())
        if (native_report['split']!='dev' or native_report['cohort']!=a.cohort
                or native_report['corpus_sha256']!=corpus_sha):
            raise ValueError('Native and independent DEV cohort differ')
        delta=metrics['weighted_nll']-native_report['metrics']['weighted_nll']
        if abs(delta)>.0002:raise AssertionError('Independent CPU/native full-panel NLL differs')
        target=a.output/'recovery'/run/'checkpoint.pt';target.parent.mkdir(parents=True)
        if shutil.disk_usage(target.parent).free<40*1024**3:raise RuntimeError('Recovery disk reserve')
        temporary=target.with_suffix('.partial');shutil.copyfile(checkpoint,temporary)
        if file_sha256(temporary)!=checkpoint_sha:raise AssertionError('Full-state recovery copy checksum differs')
        temporary.replace(target)
        loaded=torch.load(target,map_location='cpu',mmap=True,weights_only=False)
        if digest({'model':loaded['model'],'optimizer':loaded['optimizer']})!=recovered:
            raise AssertionError('Copied full-state serialized readback differs')
        del loaded
        row={'identity':run,'arm':model['arm'],'seed':model['seed'],'endpoint':model['endpoint'],
             'depth':depth,'training_real_targets':exposure,'metrics':metrics,'cpu_minus_native_nll':delta,
             'checkpoint_sha256':checkpoint_sha,'full_state_serialized_digest':recovered,
             'copy_checksum_and_full_state_readback':True,'targets':sum(s.tokens for s in scores),
             'active_seconds':time.time()-begin}
        records.append(row);write_state(a.output/'progress.json',{'records':records,'wall_seconds':time.time()-started})
        print(json.dumps({'identity':run,'cpu_minus_native_nll':delta,'recovery':'PASS'}),flush=True)
        del cpu,scores;gc.collect()
    write_state(a.output/'report.json',{'status':'PASS_DEPTH_CONNECTION_INDEPENDENT_AUDITS','records':records,
                'wall_seconds_including_waits':time.time()-started,'active_seconds':sum(r['active_seconds'] for r in records),
                'scope':'Separate FP32 implementation, full fresh DEV panel; numerical verification, not model replication. Full-state copies are on the same instance-local NVMe, not off-instance backups.'})


if __name__=='__main__':main()
