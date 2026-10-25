"""CPU preparation for a finite four-cell qualification,never a native launcher."""
import argparse
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch

from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig,future_span_targets
from compute_trainium.future_span_data import count_future_anchors
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_candidate import parameter_groups
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();started=time.time()
    out=args.root/'future-span-qualification-preparation-v1'
    public=Path('results/future_span_qualification_20261025/preparation-v1')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    parent_file=args.root/'parcae-low-depth-pilot-v1/main-plan-recovery-v2.json'
    parent=json.loads(parent_file.read_text())
    old_file=args.root/'parcae-low-depth-qualification-v1/plan.json';old=json.loads(old_file.read_text())
    config=FutureSpanConfig();model_config=ParcaeCandidateConfig(**parent['model'])
    sources={p:file_sha256(p) for p in ['scripts/prepare_future_span_qualification.py',
        'compute_trainium/future_span.py','compute_trainium/future_span_sparse.py',
        'compute_trainium/future_span_data.py','compute_trainium/parcae_candidate.py',
        'compute_trainium/parcae_fp16_candidate.py','compute_trainium/parcae_stages_candidate.py',
        'compute_trainium/parcae_optimizer_candidate.py','compute_trainium/training.py']}
    registration={'identity':'future-span-qualification-preparation-v1','registered_unix':started,
        'source_sha256':sources,'data':parent['data'],'model':parent['model'],
        'future_config':asdict(config),'head_seed':104729,'qualification_only':True,
        'parent_main_plan_sha256':file_sha256(parent_file),'old_qualification_plan_sha256':file_sha256(old_file),
        'source_windows_to_prepare':1920*512,'qualification_updates':121,'world':32,'accumulation':16,
        'cases':[{'identity':a+'-'+b,'arm':a,'objective_weight':w,'gradient_rule':'all_sampled_steps'}
                 for a in ['one','recurrent'] for b,w in [('ntp',0.),('future',.1)]],
        'check_windows':[0,1,2,31,63,1023,1920*512-1],
        'seconds_limit':600,'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,
        'native_launch_allowed':False,
        'scope':'Prepare TRAIN-only auxiliary counts,immutable release-style forward depths with '
        'full-gradient tails,and a common qualification-only initializer. No main scientific '
        'initialization,held-out scoring,native execution,optimizer updates or automatic launch.'}
    write_state(out/'registration.json',registration);write_state(public/'registration.json',registration)
    result={'status':'STOPPED_FUTURE_SPAN_QUALIFICATION_PREPARATION'}

    def guard():
        if time.time()-started>registration['seconds_limit']:raise TimeoutError('Original CPU preparation cap')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('CPU preparation host/disk reserve')

    try:
        guard();torch.set_num_threads(4)
        data=parent['data'];packing=args.root/data['packed']
        if file_sha256(packing/'manifest.json')!=data['manifest_sha256']:
            raise AssertionError('Frozen natural packing manifest changed')
        packed=PackedCorpus.from_mapped(packing,1024,23,data['corpus_sha256'])
        if packed.fingerprint!=data['packing_fingerprint']:raise AssertionError('Original packing identity differs')
        windows=registration['source_windows_to_prepare']
        counts,by_domain=count_future_anchors(packed.domains.numpy(),config,windows)
        guard();prefix=np.concatenate([np.zeros(1,dtype=np.int64),counts.cumsum(dtype=np.int64)])
        np.save(out/'anchor-counts.npy',counts);np.save(out/'anchor-prefix.npy',prefix)
        for window in registration['check_windows']:
            source=packed.tokens[window*1024:window*1024+1025].long()[None]
            real=packed.domains[window*1024:window*1024+1025][None]>0
            labels=future_span_targets(source,real,config,model_config.vocab_size)
            if int(labels.valid.sum())!=int(counts[window]):raise AssertionError('Auxiliary count and training target validity differ')
        if not np.array_equal(np.load(out/'anchor-counts.npy'),counts) or not np.array_equal(np.diff(np.load(out/'anchor-prefix.npy')),counts):
            raise AssertionError('Exact auxiliary count/prefix readback failed')
        arrays={}
        for arm,previous in [('one','fixed1'),('recurrent','poisson2')]:
            case=parent['cases'][previous+'-main-s1231']
            original=parent_file.parent/case['depth_table']
            if file_sha256(original)!=case['depth_table_sha256']:raise AssertionError('Original exogenous depth schedule changed')
            depth=np.array(np.load(original,mmap_mode='r')[:121],copy=True)
            depth[...,1]=depth[...,0]
            if depth.shape!=(121,16,32,2) or (arm=='one' and not np.all(depth==1)):
                raise AssertionError('Qualification depth geometry differs')
            path=out/(arm+'-depths.npy');np.save(path,depth)
            arrays[arm]={'file':path.name,'sha256':file_sha256(path),'shape':list(depth.shape),
                'mean_forward_depth':float(depth[...,0].mean()),'max_forward_depth':int(depth[...,0].max()),
                'all_forward_steps_differentiated':bool(np.all(depth[...,0]==depth[...,1]))}
        initial=old['initial_states']['family'];body_file=old_file.parent/initial['file']
        if old['model']!=parent['model'] or file_sha256(body_file)!=initial['sha256']:
            raise AssertionError('Qualified pinned backbone initializer changed')
        saved=torch.load(body_file,map_location='cpu',mmap=True,weights_only=False)
        if saved['qualification_only'] is not True or digest(dict(sorted(saved['model'].items())))!=initial['model_digest']:
            raise AssertionError('Original qualification-only body state differs')
        model=FutureSpanCandidate(model_config,config,head_seed=registration['head_seed'],initialize=False,fp16_body=True)
        missing=model.load_state_dict(saved['model'],strict=False)
        if set(missing.missing_keys)!={'future_lm_head.weight','future_lm_head.bias','future_positions'} or missing.unexpected_keys:
            raise AssertionError('Future initializer changed the backbone inventory')
        if digest(dict(sorted(model.prediction_state_dict().items())))!=initial['model_digest']:
            raise AssertionError('Shared original backbone initialization changed')
        groups=parameter_groups(model.named_parameters(),gate_policy='release_muon')
        if any(not any(p is h for p in groups[0]['params']) for h in model.future_lm_head.parameters()):
            raise AssertionError('Auxiliary parameters must use the explicit AdamW head group')
        full_digest=digest(dict(sorted(model.state_dict().items())))
        atomic_checkpoint(out/'initial-state.pt',{'model':model.state_dict(),'config':asdict(model_config),
            'future_config':asdict(config),'qualification_only':True,'backbone_model_seed':old['model_seed'],
            'head_seed':registration['head_seed'],'backbone_digest':initial['model_digest'],'model_digest':full_digest})
        readback=torch.load(out/'initial-state.pt',map_location='cpu',mmap=True,weights_only=False)
        if digest(dict(sorted(readback['model'].items())))!=full_digest:raise AssertionError('Qualification initializer readback differs')
        if any(file_sha256(p)!=sha for p,sha in sources.items()):raise AssertionError('Frozen preparation source changed')
        guard()
        manifest={'identity':registration['identity'],'registration_sha256':file_sha256(out/'registration.json'),
            'source_sha256':sources,'data':data,'model':parent['model'],'future_config':asdict(config),
            'qualification_only':True,'head_seed':registration['head_seed'],'backbone_model_seed':old['model_seed'],
            'state_seed':2009,'routing_seed':parent['routing_seed'],'world':32,'accumulation':16,
            'source_windows':windows,'source_real_targets':sum(packed.counts(0,windows)),
            'valid_anchors':int(prefix[-1]),'executed_anchor_slots':windows*len(config.positions),
            'valid_anchors_by_domain':dict(zip(['web','math','code'],map(int,by_domain))),
            'auxiliary_label_presentations':int(prefix[-1])*(config.horizon-1),
            'counts_file':'anchor-counts.npy','counts_sha256':file_sha256(out/'anchor-counts.npy'),
            'prefix_file':'anchor-prefix.npy','prefix_sha256':file_sha256(out/'anchor-prefix.npy'),
            'depths':arrays,'initial_file':'initial-state.pt','initial_sha256':file_sha256(out/'initial-state.pt'),
            'model_digest':full_digest,'backbone_digest':initial['model_digest'],
            'training_parameters':sum(p.numel() for p in model.parameters()),
            'deployed_parameters':model_config.parameter_count(True),
            'cases':registration['cases'],'native_launch_allowed':False}
        write_state(out/'manifest.json',manifest);write_state(public/'manifest.json',manifest)
        result={'status':'COMPLETE_CPU_FUTURE_SPAN_QUALIFICATION_PREPARATION',
            'manifest_sha256':file_sha256(out/'manifest.json'),'source_windows':windows,
            'source_real_targets':manifest['source_real_targets'],'valid_anchors':int(prefix[-1]),
            'valid_anchors_by_domain':manifest['valid_anchors_by_domain'],
            'auxiliary_label_presentations':manifest['auxiliary_label_presentations'],
            'head_and_backbone_initialization_readback_exact':True,
            'global_anchor_prefix_readback_exact':True,'native_launch_allowed':False}
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(assigned_chips=0,optimizer_updates=0,test_accessed=False,
            wall_seconds=time.time()-started,scope=registration['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}),flush=True)


if __name__=='__main__':main()
