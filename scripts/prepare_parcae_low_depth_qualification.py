"""Prepare the owner's three low-depth arms without training or model scoring."""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict,replace
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch

from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig
from compute_trainium.parcae_training_candidate import poisson_tail
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint,file_sha256
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN,reference_config,PlainTiedHead


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    args=parser.parse_args()
    out=args.root/'parcae-low-depth-qualification-v1'
    public=Path('results/parcae_low_depth_qualification_20261024')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    started=time.time()
    torch.set_num_threads(8)
    if subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(args.upstream),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean pinned upstream source required')
    sys.path.insert(0,str(args.upstream))
    config=ParcaeCandidateConfig(vocab_size=32000)
    source=json.loads((args.root/'parcae-native-global-v1/plan.json').read_text())
    files=['scripts/prepare_parcae_low_depth_qualification.py','compute_trainium/parcae_training_candidate.py',
        'compute_trainium/parcae_candidate.py','scripts/qualify_parcae_candidate_cpu.py']
    plan={'identity':'parcae-low-depth-qualification-v1','registered_unix':started,
        'source_sha256':{p:file_sha256(p) for p in files},'upstream_commit':PIN,
        'model':asdict(config),'model_seed':1009,'state_seed':2009,'context':1024,
        'source_corpus_sha256':source['source_corpus_sha256'],
        'packed_manifest_sha256':source['packed_manifest_sha256'],'packing_fingerprint':source['packing_fingerprint'],
        'data_seed':23,'routing_seed':271828,'world':32,'batch':1,'accumulations':[8,16],
        'arms':['ordinary','fixed1','poisson2'],'poisson_mean':2,'backward_cap':1,
        'expected_clamped_poisson_depth':2+float(np.exp(-2)),
        'warmup_updates':20,'measured_updates':100,'updates_through_probe':121,
        'qualification_peak_lr':1e-4,'clip':1.,'qualification_only':True,'test_accessed':False,
        'scope':'Three-arm low-depth engine preparation after owner steering. Shared backbone tensors are identical at initialization; family fixed1/Poisson2 have exactly the same parameters and complete weights. Ordinary omits injection/readout machinery and has fewer parameters. CPU noise/depth fixtures only, no native/scientific training.'}
    write_state(out/'plan.json',plan);write_state(public/'preparation-plan.json',plan)
    records=[]

    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Low-depth CPU preparation reserve')
        if time.time()-started>1200:raise TimeoutError('CPU preparation twenty-minute cap')
    try:
        guard();torch.manual_seed(1009)
        family=replace(reference_config(None,config,True),mean_recurrence=2,mean_backprop_depth=1).construct_model().train()
        family_state=family.state_dict()
        atomic_checkpoint(out/'initial-state.pt',{'model':family_state,'config':asdict(config),
            'qualification_only':True,'recurrent':True,'model_seed':1009,'upstream_commit':PIN})
        ordinary=ParcaeCandidate(config,recurrent=False,initialize=False)
        mapped={}
        for name in ordinary.state_dict():
            source_name=name
            if name.startswith('transformer.h.'):
                pieces=name.split('.')
                layer=int(pieces[2])
                part,offset=('prelude',0) if layer<2 else ('core_block',2) if layer<4 else ('coda',4)
                source_name='.'.join(['transformer',part,str(layer-offset),*pieces[3:]])
            mapped[name]=family_state[source_name]
        ordinary.load_state_dict(mapped)
        # Independently verify the ordinary model's inventory/load using upstream.
        import parcae_lm.models.gpt.gpt as gpt
        gpt.LinearCrossEntropyLoss=PlainTiedHead
        reference=reference_config(None,config,False).construct_model()
        reference.load_state_dict(ordinary.state_dict(),strict=True)
        if digest(dict(sorted(reference.state_dict().items())))!=digest(dict(sorted(ordinary.state_dict().items()))):
            raise AssertionError('Shared-backbone ordinary load differs from upstream')
        atomic_checkpoint(out/'initial-state-ordinary.pt',{'model':ordinary.state_dict(),'config':asdict(config),
            'qualification_only':True,'recurrent':False,'model_seed':1009,'upstream_commit':PIN})
        plan['initial_states']={}
        for label,model,path in [('ordinary',ordinary,'initial-state-ordinary.pt'),('family',family,'initial-state.pt')]:
            count=sum(p.numel() for p in model.parameters())
            if count!=config.parameter_count(label=='family'):raise AssertionError('Parameter inventory differs')
            plan['initial_states'][label]={'file':path,'sha256':file_sha256(out/path),
                'model_digest':digest(dict(sorted(model.state_dict().items()))),'parameters':count}
        plan['all_shared_backbone_tensors_cloned']=all(torch.equal(v,mapped[n]) for n,v in ordinary.state_dict().items())
        if not plan['all_shared_backbone_tensors_cloned']:raise AssertionError('Backbone initialization differs')
        for accumulation in [8,16]:
            for arm in ['ordinary','fixed1','poisson2']:
                guard()
                depths=np.ones((121,accumulation,32,2),dtype=np.uint16)
                if arm=='poisson2':
                    for update in range(121):
                        for micro in range(accumulation):
                            step=update*accumulation+micro+1
                            for rank in range(32):depths[update,micro,rank]=poisson_tail(2,1,step,rank)
                    for step in [1,2,17,31]:
                        for rank in [0,1,7,31]:
                            family.step=(514229+step)*(rank+1)-514229
                            prefix,tail=family._sample_batch_depths(1)
                            if (int(prefix[0]+tail[0]),int(tail[0]))!=poisson_tail(2,1,step,rank):
                                raise AssertionError('Low-depth release sampler mismatch')
                file=out/f'{arm}-a{accumulation}-depths.npy';np.save(file,depths)
                records.append({'arm':arm,'accumulation':accumulation,'depth_table_sha256':file_sha256(file),
                    'initial_kind':'ordinary' if arm=='ordinary' else 'family',
                    'minimum_depth':int(depths[...,0].min()),'maximum_depth':int(depths[...,0].max()),
                    'mean_depth':float(depths[...,0].mean()),'mean_backward_tail':float(depths[...,1].mean())})
        if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Preparation source changed')
        plan['cases']=records
        write_state(out/'plan.json',plan);write_state(public/'preparation-plan.json',plan)
        result={'status':'COMPLETE_PARCAE_TRAINING_QUALIFICATION_PREPARATION','identity':plan['identity'],
            'initial_states':plan['initial_states'],'records':records,'all_shared_backbone_tensors_cloned':True,
            'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,
            'wall_seconds_after_imports':time.time()-started,'scope':plan['scope']}
        del family,ordinary,reference;gc.collect()
    except BaseException as error:
        result={'status':'STOPPED_LOW_DEPTH_CPU_PREPARATION','records':records,'error':str(error),
            'assigned_chips':0,'optimizer_updates':0,'wall_seconds_after_imports':time.time()-started}
        raise
    finally:
        write_state(out/'cpu-report.json',result);write_state(public/'cpu-report.json',result)


if __name__=='__main__':main()
