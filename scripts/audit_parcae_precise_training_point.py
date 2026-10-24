"""Independent upstream CPU gradient at the trained qualification checkpoint."""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_training_candidate import gradient_comparison
from compute_trainium.parcae_training_candidate import configure_optimizer
from compute_trainium.parcae_recovery_check import moments
from compute_trainium.parcae_update_qualification import named_moments
from compute_trainium.parcae_optimizer_precise import independent_fp32_reference
from compute_trainium.parcae_optimizer_alignment import align_reference_groups
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus, atomic_checkpoint, file_sha256
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN, reference_config,PlainTiedHead
from scripts.qualify_parcae_optimizer_cpu import unwrapped_reference
from scripts.run_parcae_optimizer_native import step_comparison,step_passes
from third_party.parcae_reference import muon_adamw as reference_optimizer


def make_reference(config,recurrent):
    cfg=reference_config(None,config,recurrent)
    if recurrent:
        cfg=replace(cfg,mean_recurrence=2,mean_backprop_depth=1)
    else:
        import parcae_lm.models.gpt.gpt as gpt
        gpt.LinearCrossEntropyLoss=PlainTiedHead
    return cfg.construct_model().train()


def first_update(config,preparation,primary,native,output):
    """Use the observed raw vector to isolate clipping + optimizer arithmetic."""
    for file,key in [('initial-global-native.pt','initial_global_gradient_sha256'),
                     ('after-1-model-optimizer.pt','first_checkpoint_sha256')]:
        if file_sha256(primary/file)!=native[key]:raise AssertionError('First actual-update artifact changed')
    recurrent=native['arm']!='ordinary'
    initial=torch.load(preparation/('initial-state.pt' if recurrent else 'initial-state-ordinary.pt'),map_location='cpu',mmap=True,weights_only=False)
    model=make_reference(config,recurrent)
    model.load_state_dict(initial['model'])
    vector=torch.load(primary/'initial-global-native.pt',map_location='cpu',mmap=True,weights_only=False)
    actual=torch.load(primary/'after-1-model-optimizer.pt',map_location='cpu',mmap=True,weights_only=False)
    unwrapped_reference()
    precise_reference_step=independent_fp32_reference(reference_optimizer)
    if actual['optimizer']['recipe']['specified_muon_polar_dtype']!='fp32':
        raise AssertionError('Explicit FP32-polar recipe required')
    groups=reference_optimizer.get_muon_param_groups(
        model.named_parameters(),adamw_lr=.008,adamw_betas=(.8,.95),adamw_eps=1e-10,adamw_wd=0.,muon_lr=.008)
    groups=align_reference_groups(model,groups,actual['optimizer']['recipe']['groups'])
    optimizer=reference_optimizer.MuonAdamW(groups)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    if set(before)!=set(vector['gradients']):raise AssertionError('First-gradient parameter inventory differs')
    lookup={id(p):n for n,p in model.named_parameters()}
    if [[lookup[id(p)] for p in g['params']] for g in optimizer.param_groups]!=actual['optimizer']['recipe']['groups']:
        raise AssertionError('First-update optimizer grouping differs')
    for n,p in model.named_parameters():p.grad=vector['gradients'][n].clone()
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False))
    configure_optimizer(optimizer,0)
    original_reference_step=reference_optimizer.muon_step
    try:
        reference_optimizer.muon_step=precise_reference_step
        optimizer.step()
    finally:
        reference_optimizer.muon_step=original_reference_step
    expected={'model':{n:p.detach() for n,p in model.named_parameters()},
              'moments':named_moments(model,optimizer,reference=True)}
    right={'model':{n:actual['model'][n] for n in before},'moments':moments(actual['optimizer'])}
    result=step_comparison(before,expected,before,right)
    result.update(passes_registered_gate=step_passes(result),cpu_norm_before_clip=norm,
        independent_group_membership_validated=True,reference_registration_order_aligned_by_name=True,
        native_norm_before_clip=vector['raw_gradient_norm'],
        prescribed_cpu_optimizer_steps=1,research_optimizer_updates=0,
        scope='Same observed initial raw global gradient; independently checks CPU clipping and independent released algorithm with only polar dtype changed to FP32 against the actual first native update, including zero-gradient matrices. Full gradient correctness is checked separately at the trained point.')
    write_state(output/'first-update.json',result)
    return result


def audit(root,primary,upstream,output):
    started=time.time()
    output.mkdir(exist_ok=False)
    torch.set_num_threads(12)
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean pinned upstream CPU reference required')
    sys.path.insert(0,str(upstream))
    native=json.loads((primary/'report.json').read_text())
    if native['status']!='CAPTURED_PARCAE_TRAINING_QUALIFICATION' or native['mode']!='warm':
        raise ValueError('Completed warm qualification required')
    recurrent=native['arm']!='ordinary'
    paths={'after-120-full-state.pt':'checkpoint_sha256','probe-depths.npy':'probe_depths_sha256',
        'updated-global-native.pt':'updated_global_gradient_sha256'}
    if recurrent:paths['probe-initial-states.npy']='probe_initial_states_sha256'
    for file,key in paths.items():
        if file_sha256(primary/file)!=native[key]:
            raise AssertionError('Native trained-point artifact changed')
    preparation=root/'parcae-low-depth-qualification-v1'
    plan=json.loads((preparation/'plan.json').read_text())
    config=ParcaeCandidateConfig(**plan['model'])
    first=first_update(config,preparation,primary,native,output)
    if not first['passes_registered_gate']:
        result={'status':'FAILED_INDEPENDENT_FIRST_PARCAE_UPDATE','first_actual_update':first,
            'arm':native['arm'],'precision':native['precision'],'accumulation':native['accumulation'],
            'assigned_chips':0,'prescribed_cpu_optimizer_steps':1,'research_optimizer_updates':0,'wall_seconds':time.time()-started}
        write_state(output/'report.json',result)
        return result
    gc.collect()
    saved=torch.load(primary/'after-120-full-state.pt',map_location='cpu',mmap=True,weights_only=False)
    if digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})!=native['checkpoint_state_digest']:
        raise AssertionError('Full trained checkpoint readback differs')
    cursor=saved['cursor']
    count=32*native['accumulation']
    packed=PackedCorpus.from_mapped(root/'natural-append-6500m-v2/packed',1024,23,plan['source_corpus_sha256'])
    total=sum(packed.counts(cursor,count))
    model=make_reference(config,recurrent)
    model.load_state_dict(saved['model'],strict=True)
    model_digest=digest(dict(sorted(model.state_dict().items())))
    del saved
    noise=np.load(primary/'probe-initial-states.npy',mmap_mode='r') if recurrent else None
    depths=np.load(primary/'probe-depths.npy',mmap_mode='r')
    if (noise is not None and noise.shape!=(count,1024,config.dim)) or depths.shape!=(count,2):
        raise AssertionError('Trained global-probe shape differs')
    current_state=None
    if recurrent:model.initialize_state=lambda _:current_state
    loss_sum=0.

    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3 or min(shutil.disk_usage(output).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Trained-point CPU audit reserve')
        if time.time()-started>1800:
            raise TimeoutError('Trained-point CPU thirty-minute cap')
    with sdpa_kernel(SDPBackend.MATH):
        for index in range(count):
            guard()
            batch=packed.batch([cursor+index],0)[0]
            current_state=torch.from_numpy(np.array(noise[index:index+1],copy=True)) if recurrent else None
            n,k=map(int,depths[index])
            kwargs={'num_steps_pair':torch.tensor([n-k,k])} if recurrent else {}
            logits=model(batch.inputs,return_logits=True,**kwargs)['logits']
            value=(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none')*batch.target_mask.flatten()).sum()
            (value/total).backward()
            loss_sum+=float(value.detach())
            if (index+1)%32==0:
                write_state(output/'progress.json',{'windows':index+1,'total_windows':count,'elapsed_seconds':time.time()-started})
    gradients={n:p.grad.detach() for n,p in model.named_parameters()}
    actual=torch.load(primary/'updated-global-native.pt',map_location='cpu',mmap=True,weights_only=False)
    delta=gradient_comparison(gradients,actual['gradients'])
    nll_difference=actual['loss']-loss_sum/total
    passed=delta['passes_two_percent_gate'] and abs(nll_difference)<=.003
    if actual['real_targets']!=total or actual['source_cursor']!=cursor:
        raise AssertionError('Trained global objective source normalization differs')
    if digest(dict(sorted(model.state_dict().items())))!=model_digest:
        raise AssertionError('Independent CPU backward changed weights')
    atomic_checkpoint(output/'cpu-global.pt',{'gradients':gradients,'loss':loss_sum/total,
        'real_targets':total,'source_cursor':cursor})
    result={'status':'PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT' if passed else 'FAILED_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT',
        'arm':native['arm'],'precision':native['precision'],'accumulation':native['accumulation'],
        'source_cursor':cursor,'real_targets':total,'windows':count,'gradient':delta,
        'cpu_nll':loss_sum/total,'native_minus_cpu_nll':nll_difference,
        'checkpoint_sha256':native['checkpoint_sha256'],'full_state_readback_exact':True,
        'cpu_global_gradient_sha256':file_sha256(output/'cpu-global.pt'),
        'first_actual_update':first,
        'wall_seconds':time.time()-started,'assigned_chips':0,'prescribed_cpu_optimizer_steps':1,
        'native_optimizer_updates':0,'research_optimizer_updates':0,'test_accessed':False,
        'scope':'Independent pinned upstream FP32 complete next-global-update objective at the actual120-update qualification checkpoint. Exact supplied hidden states/depths and natural TRAIN targets. Numerical evidence, not held-out learning or throughput.'}
    write_state(output/'report.json',result)
    del model,gradients,actual
    gc.collect()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        audit(args.root,args.primary,args.upstream,args.output)
    except BaseException as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write_state(args.output/'failure.json',{'status':'STOPPED_INDEPENDENT_TRAINED_PARCAE_AUDIT','error':str(error),
            'assigned_chips':0,'prescribed_cpu_optimizer_steps_completed':None,
            'maximum_prescribed_cpu_optimizer_steps':1,'native_optimizer_updates':0,
            'research_optimizer_updates':0,'test_accessed':False,
            'scope':'Exception receipt alone does not establish whether the one CPU reference optimizer step completed; first-update.json records it if comparison finished.'})
        raise


if __name__=='__main__':main()
