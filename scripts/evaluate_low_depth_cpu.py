"""Primary FP32 scores using pinned upstream model code on consumed DEV only."""
import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.data import document_windows
from compute_trainium.metrics import DocumentScore,summarize
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_training_candidate import initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.qualify_native import digest


def window_state_key(document_id,window):
    if not document_id or type(window) is not int or window<0:raise ValueError('Stable document/window identity required')
    return int(hashlib.sha256((document_id+':'+str(window)).encode()).hexdigest()[:16],16)%(2**63-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha',required=True)
    p.add_argument('--case',required=True);p.add_argument('--cohort',choices=['tune','curve'],required=True)
    p.add_argument('--depth',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--seconds-limit',type=int,default=900)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.time()
    torch.set_num_threads(12);torch.set_num_interop_threads(1)
    if file_sha256(a.plan)!=a.plan_sha or file_sha256(a.checkpoint)!=a.checkpoint_sha:
        raise AssertionError('Frozen scientific plan/checkpoint changed')
    plan=json.loads(a.plan.read_text());case=plan['cases'][a.case]
    if plan['purpose']!='natural_low_depth_scientific_pilot':raise ValueError('Qualification states cannot be scientific readouts')
    if a.depth not in ([1] if case['arm']!='poisson2' else [1,2,3]):raise ValueError('Unregistered or unsupported evaluation depth')
    if subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean pinned independent upstream evaluator required')
    sys.path.insert(0,str(a.upstream))
    source=a.root/plan['evaluation']['corpus']
    if file_sha256(source)!=plan['evaluation']['corpus_sha256'] or plan['evaluation']['scope']!='consumed_development_only':
        raise ValueError('This pilot can score only its frozen consumed DEV panel')
    docs=sorted([d for d in json.loads(source.read_text()) if d['split']=='dev' and d['cohort']==a.cohort],key=lambda d:d['id'])
    if not docs:raise ValueError('Empty declared development cohort')
    saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if (saved['qualification_only'] is not False or saved['purpose']!=plan['purpose']
        or saved['signature']['plan_sha256']!=a.plan_sha or saved['case_id']!=a.case):
        raise ValueError('Scientific checkpoint identity/purpose differs')
    state_digest=digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})
    if state_digest!=saved['model_optimizer_digest']:raise AssertionError('Complete checkpoint readback differs')
    exposure=saved['real_targets'];work=saved['executed_matmuls'];update=saved['update']
    config=ParcaeCandidateConfig(**plan['model']);recurrent=case['arm']!='ordinary'
    model=make_reference(config,recurrent).eval();model.load_state_dict(saved['model'],strict=True)
    model_digest=digest(dict(sorted(model.state_dict().items())));del saved
    current_state=None
    if recurrent:model.initialize_state=lambda _:current_state
    scores=[];windows=0
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3:raise RuntimeError('Independent scoring memory reserve')
        if time.time()-started>a.seconds_limit:raise TimeoutError('Independent scoring deadline')
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        for index,d in enumerate(docs):
            guard();total=0.;count=0
            for window,(tokens,mask) in enumerate(document_windows(d['tokens'],1024)):
                n=int(mask.sum())
                current_state=initial_state(config,window_state_key(d['id'],window),case['state_seed'])[:,:n] if recurrent else None
                kwargs={'num_steps_pair':torch.tensor([a.depth-1,1])} if recurrent else {}
                logits=model(tokens[None,:n],return_logits=True,**kwargs)['logits'][0]
                if logits.shape!=(n,config.vocab_size) or not torch.isfinite(logits).all():
                    raise AssertionError('Independent source logits shape/finiteness failed')
                for start in range(0,n,128):
                    end=min(start+128,n);values=logits[start:end].double();target=tokens[start+1:end+1]
                    total+=float((values.logsumexp(-1)-values.gather(-1,target[:,None]).squeeze(-1)).sum())
                count+=n;windows+=1
                del logits
            if count!=len(d['tokens']):raise AssertionError('Independent evaluator discarded a source target')
            scores.append(DocumentScore(d['domain'],d['id'],d['group'],total,count))
            if (index+1)%100==0:
                write_state(a.output/'progress.json',{'documents':index+1,'windows':windows,'wall_seconds':time.time()-started})
    if digest(dict(sorted(model.state_dict().items())))!=model_digest:raise AssertionError('Independent scoring changed weights')
    report={'status':'COMPLETE_INDEPENDENT_LOW_DEPTH_DEVELOPMENT_SCORE','case_id':a.case,'arm':case['arm'],
        'seed':case['seed'],'depth':a.depth,'cohort':a.cohort,'split':'dev','training_update':update,
        'training_real_targets':exposure,'training_executed_matmuls':work,'parameters':case['parameters'],
        'metrics':summarize(scores),'documents':len(scores),'real_targets':sum(r.tokens for r in scores),
        'windows':windows,'checkpoint_sha256':a.checkpoint_sha,'full_state_readback_exact':True,
        'full_state_digest':state_digest,'source_corpus_sha256':plan['evaluation']['corpus_sha256'],
        'upstream_commit':PIN,'model_unchanged':True,'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,
        'wall_seconds_after_imports':time.time()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'scope':'Primary FP32 independent upstream prediction on frozen previously consumed DEV. No fresh TEST/confirmation, training-seed significance or self-generated/teacher labels. Raw texts, tokens and model/optimizer state stay private.'}
    write_state(a.output/'report.json',report)
    # Keep per-document paired results private; publish only aggregate metrics.
    write_state(a.output/'document-scores.json',{'records':[{'id':s.document,'domain':s.domain,'group':s.group,
        'nll_sum':s.nll_sum,'real_targets':s.tokens} for s in scores]})
    print(json.dumps({'case':a.case,'nll':report['metrics']['weighted_nll'],'seconds':time.time()-started}),flush=True)
    del model;gc.collect()


if __name__=='__main__':main()
