"""Fresh random initializers and append-compatible TRAIN ledgers;no fit launcher."""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch

from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig,future_span_targets
from compute_trainium.future_span_data import count_future_anchors
from compute_trainium.future_span_scientific_state import matrix_work
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_work_accounting import muon_update_matmuls
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.qualify_native import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();started=time.time()
    out=args.root/'future-span-study-preparation-v1'
    public=Path('results/future_span_scientific_preparation_20261025')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    parent_file=args.root/'parcae-low-depth-pilot-v1/main-plan-recovery-v2.json'
    parent=json.loads(parent_file.read_text());config=ParcaeCandidateConfig(**parent['model']);span=FutureSpanConfig()
    seeds=[4127,5231,7283]
    previous_seeds=set()
    def inspect(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k in ['seed','model_seed','backbone_model_seed'] and type(v) is int:previous_seeds.add(v)
                elif isinstance(v,(dict,list)):inspect(v)
        elif isinstance(obj,list):
            for v in obj:inspect(v)
    for path in Path('results').rglob('*.json'):inspect(json.loads(path.read_text()))
    if set(seeds)&previous_seeds:raise ValueError('Proposed research initializer seed already appears in published results')
    sources={p:file_sha256(p) for p in ['scripts/prepare_future_span_science.py',
        'compute_trainium/future_span.py','compute_trainium/future_span_data.py',
        'compute_trainium/future_span_scientific_state.py','compute_trainium/parcae_scientific_schedule.py',
        'compute_trainium/parcae_scientific_state.py','compute_trainium/parcae_candidate.py',
        'compute_trainium/parcae_fp16_candidate.py','compute_trainium/parcae_stages_candidate.py',
        'compute_trainium/parcae_work_accounting.py','compute_trainium/training.py',
        'scripts/audit_parcae_precise_training_point.py','scripts/qualify_parcae_candidate_cpu.py']}
    registration={'identity':'future-span-study-preparation-v1','registered_unix':started,
        'source_sha256':sources,'parent_plan_sha256':file_sha256(parent_file),'data':parent['data'],
        'model':asdict(config),'future_config':asdict(span),'seeds':seeds,
        'published_seed_fields_checked':len(previous_seeds),'provisioned_updates':4096,'depth_rows':4097,
        'proposed_lr_grid':[.008,.016,.032],'scope':'Fresh untrained random weights and TRAIN-only '
        'source/auxiliary/matrix ledgers. Seed freshness checked in published result manifests. '
        'No scientific run registration,quality score,native work,automatic launch,TEST or alpha use.',
        'seconds_limit':600,'scientific_launch_allowed':False,'assigned_chips':0,'optimizer_updates':0,'test_accessed':False}
    write_state(out/'registration.json',registration);write_state(public/'registration.json',registration)
    result={'status':'STOPPED_FUTURE_SPAN_SCIENTIFIC_PREPARATION'};initials={};tables={}
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (available<544*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3
                or time.time()-started>registration['seconds_limit']):raise RuntimeError('Fresh preparation resource/cost cap')
    try:
        guard();torch.set_num_threads(8)
        upstream=args.root/'upstream-parcae-69284c1'
        if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
                or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):
            raise ValueError('Clean pinned upstream initializer required')
        sys.path.insert(0,str(upstream));data=parent['data'];packing=args.root/data['packed']
        if file_sha256(packing/'manifest.json')!=data['manifest_sha256']:raise AssertionError('Frozen packing manifest changed')
        packed=PackedCorpus.from_mapped(packing,1024,23,data['corpus_sha256'])
        if packed.fingerprint!=data['packing_fingerprint']:raise AssertionError('Natural source packing differs')
        windows=registration['depth_rows']*512
        counts,by_domain=count_future_anchors(packed.domains.numpy(),span,windows)
        prefix=np.concatenate([np.zeros(1,dtype=np.int64),counts.cumsum(dtype=np.int64)])
        np.save(out/'anchor-counts.npy',counts);np.save(out/'anchor-prefix.npy',prefix)
        if not np.array_equal(np.diff(np.load(out/'anchor-prefix.npy')),counts):raise AssertionError('Global auxiliary denominator readback differs')
        for window in [0,1,31,511,61439,windows-1]:
            source=packed.tokens[window*1024:window*1024+1025].long()[None]
            real=packed.domains[window*1024:window*1024+1025][None]>0
            if int(future_span_targets(source,real,span,config.vocab_size).valid.sum())!=int(counts[window]):
                raise AssertionError('Independent future-span mask/count mismatch')
        optimizer_ops=None
        for seed in seeds:
            guard();torch.manual_seed(seed);reference=make_reference(config,True)
            body_digest=digest(dict(sorted(reference.state_dict().items())))
            model=FutureSpanCandidate(config,span,head_seed=104729+seed,initialize=False,fp16_body=True)
            missing=model.load_state_dict(reference.state_dict(),strict=False)
            if set(missing.missing_keys)!={'future_lm_head.weight','future_lm_head.bias','future_positions'} or missing.unexpected_keys:
                raise AssertionError('New head changed the independent body initializer')
            if digest(dict(sorted(model.prediction_state_dict().items())))!=body_digest:raise AssertionError('Backbone initial weights changed')
            state_digest=digest(dict(sorted(model.state_dict().items())))
            path=out/f'initial-s{seed}.pt'
            atomic_checkpoint(path,{'model':model.state_dict(),'seed':seed,'head_seed':104729+seed,
                'config':asdict(config),'future_config':asdict(span),'qualification_only':False,
                'purpose':'fresh_random_future_span_scientific_initializer','model_digest':state_digest,
                'upstream_commit':PIN,'optimizer_updates':0})
            readback=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            if digest(dict(sorted(readback['model'].items())))!=state_digest:raise AssertionError('Fresh body/head initializer readback differs')
            initials[f's{seed}-family']={'file':path.name,'sha256':file_sha256(path),'model_digest':state_digest,
                'seed':seed,'head_seed':104729+seed,'backbone_digest':body_digest,
                'training_parameters':sum(p.numel() for p in model.parameters()),'deployed_parameters':config.parameter_count(True)}
            current_ops=muon_update_matmuls(model.named_parameters(),gate_policy='release_muon')
            if optimizer_ops is not None and current_ops!=optimizer_ops:raise AssertionError('Initializer seeds changed optimizer inventory')
            optimizer_ops=current_ops
            del reference,model,readback;gc.collect()
        qualified_dir=args.root/'future-span-qualification-preparation-v1'
        qualified=json.loads((qualified_dir/'manifest.json').read_text())
        for arm,previous in [('one','fixed1'),('recurrent','poisson2')]:
            guard();case=parent['cases'][previous+'-main-s1231'];file=parent_file.parent/case['depth_table']
            if file_sha256(file)!=case['depth_table_sha256']:raise AssertionError('Exogenous depth source changed')
            depth=np.array(np.load(file,mmap_mode='r')[:registration['depth_rows']],copy=True);depth[...,1]=depth[...,0]
            if depth.shape!=(4097,16,32,2):raise ValueError('Provision enough source/depth rows without wrapping')
            q=qualified['depths'][arm]
            if file_sha256(qualified_dir/q['file'])!=q['sha256'] or not np.array_equal(depth[:121],np.load(qualified_dir/q['file'])):
                raise AssertionError('Scientific prefix differs from hardware-qualified full-gradient depth tables')
            work=matrix_work(config,span,depth,world=32,optimizer_per_replica=optimizer_ops)
            if sum(map(int,work))>np.iinfo(np.int64).max:raise OverflowError('Complete matrix ledger overflows signed64')
            depth_file=out/(arm+'-depths.npy');work_file=out/(arm+'-work.npy');np.save(depth_file,depth);np.save(work_file,work)
            milestones={}
            for label,updates in [('tuning',576),('main',1920),('provisioned',4096)]:
                milestones[label]={'updates':updates,'real_targets':sum(packed.counts(0,updates*512)),
                    'valid_anchors':int(prefix[updates*512]),'estimated_dense_matrix_ops':int(work[:updates].sum()),
                    'cooldown_start_update':updates//2,'cooldown_start_targets':sum(packed.counts(0,(updates//2)*512))}
            tables[arm]={'depth_table':depth_file.name,'depth_table_sha256':file_sha256(depth_file),
                'work_table':work_file.name,'work_table_sha256':file_sha256(work_file),'milestones':milestones,
                'maximum_forward_depth':int(depth[...,0].max()),'mean_forward_depth':float(depth[...,0].mean())}
        guard()
        if any(file_sha256(p)!=sha for p,sha in sources.items()):raise AssertionError('Frozen preparation source changed')
        manifest={'identity':registration['identity'],'registration_sha256':file_sha256(out/'registration.json'),
            'source_sha256':sources,'upstream_commit':PIN,'model':asdict(config),'future_config':asdict(span),
            'data':data,'initial_states':initials,'tables':tables,'gradient_rule':'all_sampled_steps','routing_seed':parent['routing_seed'],
            'auxiliary_data':{'directory':out.name,'counts_file':'anchor-counts.npy','counts_sha256':file_sha256(out/'anchor-counts.npy'),
                'prefix_file':'anchor-prefix.npy','prefix_sha256':file_sha256(out/'anchor-prefix.npy'),
                'source_windows':windows,'valid_anchors':int(prefix[-1]),'valid_anchors_by_domain':dict(zip(['web','math','code'],map(int,by_domain)))},
            'optimizer_matrix_ops_per_replica':optimizer_ops,'proposed_lr_grid':registration['proposed_lr_grid'],
            'scientific_launch_allowed':False,'scope':registration['scope']}
        write_state(out/'manifest.json',manifest);write_state(public/'manifest.json',manifest)
        result={'status':'COMPLETE_FRESH_FUTURE_SPAN_SCIENTIFIC_PREPARATION','manifest_sha256':file_sha256(out/'manifest.json'),
            'fresh_seeds':seeds,'all_initialization_readbacks_exact':True,'no_qualification_weights_reused':True,
            'auxiliary_prefix_readback_exact':True,'hardware_depth_prefixes_identical':True,'scientific_launch_allowed':False}
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(assigned_chips=0,optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,scope=registration['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}),flush=True)


if __name__=='__main__':main()
