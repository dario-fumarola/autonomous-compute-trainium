"""Independent prefix-only FP32 development scores with the auxiliary head removed."""
import argparse
import gc
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.data import document_windows
from compute_trainium.future_span_evaluation import prediction_state,window_state
from compute_trainium.future_span_scientific_state import SCIENCE,signature
from compute_trainium.metrics import DocumentScore,summarize
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_scientific_state import schedule_identity
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.qualify_native import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--checkpoint-sha',required=True)
    parser.add_argument('--case',required=True);parser.add_argument('--cohort',choices=['tune','curve'],required=True)
    parser.add_argument('--depth',type=int,choices=[1,2,3],required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True);parser.add_argument('--seconds-limit',type=int,default=1200)
    args=parser.parse_args();started=time.time();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(12);torch.set_num_interop_threads(1)
    if file_sha256(args.plan)!=args.plan_sha or file_sha256(args.checkpoint)!=args.checkpoint_sha:
        raise AssertionError('Scientific plan/checkpoint changed')
    plan=json.loads(args.plan.read_text());case=plan['cases'][args.case]
    if any(file_sha256(p)!=sha for p,sha in plan['source_sha256'].items()):
        raise AssertionError('Frozen scientific/readout implementation changed')
    if (plan['purpose']!=SCIENCE or plan['evaluation']['scope']!='consumed_development_only'
            or plan['evaluation']['corpus']!='variable-depth-fresh-v1/dev.json'):
        raise ValueError('This discovery evaluator cannot open fresh confirmation or TEST')
    if case['arm']=='one' and args.depth!=1:raise ValueError('One-pass model has only its trained depth-one exit')
    if (subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
            or subprocess.check_output(['git','-C',str(args.upstream),'status','--porcelain'],text=True).strip()):
        raise ValueError('Clean pinned independent upstream evaluator required')
    sys.path.insert(0,str(args.upstream))
    source=args.root/plan['evaluation']['corpus']
    if file_sha256(source)!=plan['evaluation']['corpus_sha256']:raise AssertionError('Declared consumed panel changed')
    docs=sorted([d for d in json.loads(source.read_text()) if d['split']=='dev' and d['cohort']==args.cohort],key=lambda d:d['id'])
    if not docs:raise ValueError('Empty declared consumed-development cohort')
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if (saved['qualification_only'] is not False or saved['purpose']!=SCIENCE or saved['case_id']!=args.case
            or saved['signature']!=signature(args.plan_sha,plan,case) or saved['schedule']!=schedule_identity(case)):
        raise ValueError('Scientific checkpoint identity/objective/schedule differs')
    state_digest=digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})
    if state_digest!=saved['model_optimizer_digest']:raise AssertionError('Independent complete-state readback differs')
    cursor=saved['cursor'];update=saved['update']
    if not 0<update<=case['total_updates'] or cursor!=update*512:raise ValueError('Scientific source cursor/update differs')
    data=plan['data'];packing=args.root/data['packed']
    metadata=json.loads((packing/'manifest.json').read_text())
    if file_sha256(packing/'manifest.json')!=data['manifest_sha256']:raise AssertionError('Packing manifest changed')
    counts_file=packing/'prefix_counts.npy'
    if file_sha256(counts_file)!=metadata['files']['prefix_counts.npy']:raise AssertionError('Original source prefix counts changed')
    counts=np.load(counts_file,mmap_mode='r');domains=list(map(int,counts[cursor]-counts[0]))
    work_file=args.plan.parent/case['work_table'];aux=plan['auxiliary_data'];prefix_file=args.root/aux['directory']/aux['prefix_file']
    if file_sha256(work_file)!=case['work_table_sha256'] or file_sha256(prefix_file)!=aux['prefix_sha256']:
        raise AssertionError('Matrix or future-anchor counter table changed')
    work=int(np.load(work_file,mmap_mode='r')[:update].sum());anchors=int(np.load(prefix_file,mmap_mode='r')[cursor])
    if (domains!=saved['domain_targets'] or sum(domains)!=saved['real_targets']
            or work!=saved['estimated_dense_matrix_ops'] or anchors!=saved['valid_anchors']):
        raise AssertionError('Independent source/domain/matrix/auxiliary counters differ')
    exposure=saved['real_targets'];config=ParcaeCandidateConfig(**plan['model'])
    model=make_reference(config,True).eval();model.load_state_dict(prediction_state(saved['model']),strict=True);del saved
    before=digest(dict(sorted(model.state_dict().items())));current=None;model.initialize_state=lambda _:current
    scores=[];windows=0
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (available<544*1024**3 or min(shutil.disk_usage(args.output).free,shutil.disk_usage('/').free)<35*1024**3
                or time.time()-started>args.seconds_limit):raise RuntimeError('Independent future readout reserve/cap')
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        for index,d in enumerate(docs):
            guard();total=0.;count=0
            for window,(tokens,mask) in enumerate(document_windows(d['tokens'],1024)):
                n=int(mask.sum());current=window_state(config,window,case['state_seed'],n)
                logits=model(tokens[None,:n],return_logits=True,num_steps_pair=torch.tensor([args.depth-1,1]))['logits'][0]
                if logits.shape!=(n,config.vocab_size) or not torch.isfinite(logits).all():raise AssertionError('Independent logits shape/finiteness failed')
                for start in range(0,n,128):
                    stop=min(start+128,n);values=logits[start:stop].double();target=tokens[start+1:stop+1]
                    total+=float((values.logsumexp(-1)-values.gather(-1,target[:,None]).squeeze(-1)).sum())
                count+=n;windows+=1;del logits
            if count!=len(d['tokens']):raise AssertionError('Independent readout discarded real source targets')
            scores.append(DocumentScore(d['domain'],d['id'],d['group'],total,count))
            if (index+1)%100==0:write_state(args.output/'progress.json',{'documents':index+1,'windows':windows,'wall_seconds':time.time()-started})
    if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('Readout changed model weights')
    report={'status':'COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE','case_id':args.case,'cell':case['cell'],
        'arm':case['arm'],'objective_weight':case['objective_weight'],'seed':case['seed'],'depth':args.depth,
        'cohort':args.cohort,'split':'dev','training_update':update,'training_real_targets':exposure,
        'training_valid_anchors':anchors,'estimated_training_dense_matrix_ops':work,
        'training_parameters':case['training_parameters'],'deployed_parameters':config.parameter_count(True),
        'metrics':summarize(scores),'documents':len(scores),'real_targets':sum(s.tokens for s in scores),'windows':windows,
        'checkpoint_sha256':args.checkpoint_sha,'full_state_readback_exact':True,'full_state_digest':state_digest,
        'head_removed':True,'model_unchanged':True,'source_corpus_sha256':plan['evaluation']['corpus_sha256'],
        'upstream_commit':PIN,'latent_noise_policy':'Original zero-based window position only;fixed1024 draw before prefix slicing;no document/content/length key.',
        'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,'wall_seconds':time.time()-started,
        'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'scope':'Independent FP32 causal development readout with the training-only head removed. Consumed panel only,no fresh confirmation,TEST access or seed-significance claim.'}
    write_state(args.output/'report.json',report)
    write_state(args.output/'document-scores.json',{'records':[{'id':s.document,'domain':s.domain,'group':s.group,
        'nll_sum':s.nll_sum,'real_targets':s.tokens} for s in scores]})
    print(json.dumps({'case':args.case,'depth':args.depth,'nll':report['metrics']['weighted_nll']}),flush=True)
    del model;gc.collect()


if __name__=='__main__':main()
