"""Prepare matched release-initialized weights, sampler tables and CPU probes."""
import argparse
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
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from compute_trainium.depth_mixture import routed_indices
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_training_candidate import initial_state, poisson_tail
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus, atomic_checkpoint, file_sha256
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN, reference_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    args = parser.parse_args()
    started = time.time()
    out = args.root/'parcae-training-qualification-v1'
    public = Path('results/parcae_training_qualification_20261024')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    if subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(args.upstream),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean pinned upstream initializer required')
    sys.path.insert(0,str(args.upstream))
    reference_root = args.root/'parcae-native-global-v1'
    reference_plan = json.loads((reference_root/'plan.json').read_text())
    config = ParcaeCandidateConfig(vocab_size=32000)
    paths = ['scripts/prepare_parcae_training_qualification.py',
        'compute_trainium/parcae_training_candidate.py','scripts/qualify_parcae_candidate_cpu.py',
        'compute_trainium/parcae_candidate.py']
    plan = {'identity':'parcae-training-qualification-v1','registered_unix':started,
        'upstream_commit':PIN,'model':asdict(config),'model_seed':1009,'state_seed':2009,
        'source_sha256':{p:file_sha256(p) for p in paths},
        'source_corpus_sha256':reference_plan['source_corpus_sha256'],
        'packed_manifest_sha256':reference_plan['packed_manifest_sha256'],
        'packing_fingerprint':reference_plan['packing_fingerprint'],
        'context':1024,'world':32,'batch':1,'accumulations':[8,16],
        'warmup_updates':20,'measured_updates':100,'updates_through_probe':121,
        'arms':['fixed1','poisson8'],'poisson_mean':8,'backward_cap':4,
        'seed_rule':'((514229+one_based_microstep)*(rank+1)) modulo(2^31-1)',
        'qualification_peak_lr':1e-4,'clip':1.,'data_seed':23,'routing_seed':271828,
        'qualification_only':True,'test_accessed':False,'cpu_seconds_limit':1200,
        'scope':'Unperturbed pinned release initialization shared across fixed-one-pass and Poisson8/tail4 family. Numerical/training-throughput qualification only. No scientific fit or future reuse of trained qualification weights.'}
    write_state(out/'plan.json',plan);write_state(public/'preparation-plan.json',plan)
    records=[]

    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('CPU training-preparation reserve')
        if time.time()-started>1200:
            raise TimeoutError('Twenty-minute CPU preparation cap')
    try:
        guard()
        packing=args.root/'natural-append-6500m-v2/packed'
        packed=PackedCorpus.from_mapped(packing,1024,23,plan['source_corpus_sha256'])
        if packed.fingerprint!=plan['packing_fingerprint']:
            raise AssertionError('Natural source packing changed')
        torch.manual_seed(1009)
        model=reference_config(None,config,True).construct_model().train()
        if sum(p.numel() for p in model.parameters())!=config.parameter_count(True):
            raise AssertionError('Independent family parameter inventory differs')
        atomic_checkpoint(out/'initial-state.pt',{'model':model.state_dict(),'config':asdict(config),
            'qualification_only':True,'upstream_commit':PIN,'model_seed':1009,'recurrent':True})
        initial_digest=digest(dict(sorted(model.state_dict().items())))
        plan.update(initial_checkpoint_sha256=file_sha256(out/'initial-state.pt'),
            initial_model_digest=initial_digest,parameters=config.parameter_count(True))
        for accumulation in [8,16]:
            for arm in ['fixed1','poisson8']:
                guard()
                depths=np.ones((121,accumulation,32,2),dtype=np.uint16)
                if arm=='poisson8':
                    for update in range(121):
                        for micro in range(accumulation):
                            step=update*accumulation+micro+1
                            for rank in range(32):
                                depths[update,micro,rank]=poisson_tail(8,4,step,rank)
                    # Check the actual released per-sequence method, including
                    # rank seeding, without changing distributed global state.
                    for step in [1,2,17,31]:
                        for rank in [0,1,7,31]:
                            model.step=(514229+step)*(rank+1)-514229
                            prefix,tail=model._sample_batch_depths(1)
                            if (int(prefix[0]+tail[0]),int(tail[0]))!=poisson_tail(8,4,step,rank):
                                raise AssertionError('Release sampler mismatch')
                table=out/f'{arm}-a{accumulation}-depths.npy'
                np.save(table,depths)
                index=routed_indices(0,0,0,32,1,accumulation,271828)[0]
                batch=packed.batch([index],0)[0]
                state=initial_state(config,index,2009)
                model.initialize_state=lambda _:state
                n,k=map(int,depths[0,0,0])
                model.zero_grad(set_to_none=True)
                with sdpa_kernel(SDPBackend.MATH):
                    logits=model(batch.inputs,return_logits=True,num_steps_pair=torch.tensor([n-k,k]))['logits']
                    value=(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none')*batch.target_mask.flatten()).sum()/batch.target_mask.sum()
                    value.backward()
                vector=out/f'{arm}-a{accumulation}-initial-local.pt'
                atomic_checkpoint(vector,{'gradients':{name:p.grad.detach().clone() for name,p in model.named_parameters()},
                    'loss':float(value.detach()),'source_window':index,'depth':n,'backward_cap':k})
                records.append({'arm':arm,'accumulation':accumulation,'depth_table_sha256':file_sha256(table),
                    'initial_local_reference_sha256':file_sha256(vector),
                    'minimum_depth':int(depths[...,0].min()),'maximum_depth':int(depths[...,0].max()),
                    'mean_depth':float(depths[...,0].mean()),'mean_backward_tail':float(depths[...,1].mean()),
                    'sampler_matches_pinned_release':arm=='poisson8','local_source_window':index,
                    'local_depth':n,'local_backward_cap':k,'local_numerical_nll':float(value.detach())})
                write_state(public/'cpu-progress.json',{'records':records})
        if digest(dict(sorted(model.state_dict().items())))!=initial_digest:
            raise AssertionError('CPU gradient/sampler preparation changed weights')
        if any(file_sha256(p)!=sha for p,sha in plan['source_sha256'].items()):
            raise AssertionError('CPU training preparation source changed')
        plan['cases']=records
        write_state(out/'plan.json',plan);write_state(public/'preparation-plan.json',plan)
        result={'status':'COMPLETE_PARCAE_TRAINING_QUALIFICATION_PREPARATION','records':records,
            'initial_checkpoint_sha256':plan['initial_checkpoint_sha256'],'parameters':plan['parameters'],
            'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,
            'wall_seconds_after_imports':time.time()-started,'scope':plan['scope']}
    except BaseException as error:
        result={'status':'STOPPED_PARCAE_TRAINING_QUALIFICATION_PREPARATION','records':records,
            'error':str(error),'assigned_chips':0,'optimizer_updates':0,'wall_seconds_after_imports':time.time()-started}
        raise
    finally:
        write_state(out/'cpu-report.json',result);write_state(public/'cpu-report.json',result)


if __name__=='__main__':main()
