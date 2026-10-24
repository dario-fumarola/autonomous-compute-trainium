"""Prepare independent full-window/global CPU references; no native work."""
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
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig
from compute_trainium.depth_mixture import routed_indices
from compute_trainium.training import PackedCorpus, atomic_checkpoint, file_sha256
from compute_trainium.queue import write_state
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import reference_config, PlainTiedHead, PIN


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    a=p.parse_args()
    started=time.time()
    out=a.root/'parcae-native-global-v1'
    out.mkdir(exist_ok=False)
    public=Path('results/parcae_native_global_20261024')
    public.mkdir(exist_ok=True)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    deadline=started+2700
    def guard():
        fields={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        if fields['MemAvailable']<544*1024**3:
            raise RuntimeError('CPU preparation preserves extra32GiB above the shared memory floor')
        if time.time()>deadline:
            raise TimeoutError('CPU reference45-minute deadline')
        if shutil.disk_usage(out).free<42*1024**3 or shutil.disk_usage(public).free<35*1024**3:
            raise RuntimeError('CPU reference disk reserve')
    guard()
    revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'],text=True).strip()
    if revision!=PIN or dirty:
        raise ValueError('Clean pinned upstream code required')
    sys.path.insert(0,str(a.upstream))
    c=ParcaeCandidateConfig(vocab_size=32000)
    corpus=a.root/'natural-append-6500m-v2/source.json'
    packing=a.root/'natural-append-6500m-v2/packed'
    corpus_sha=file_sha256(corpus)
    paths=['scripts/prepare_parcae_native_reference.py','scripts/qualify_parcae_candidate_cpu.py',
           'compute_trainium/parcae_candidate.py','compute_trainium/parcae_stages_candidate.py',
           'compute_trainium/stable_recurrence_candidate.py','docs/PARCAE_NATIVE_GRADIENT_PROTOCOL_20261024.md']
    plan={'identity':'parcae-native-global-v1','registered_unix':time.time(),'upstream_commit':PIN,
        'source_sha256':{p:file_sha256(p) for p in paths},'model':asdict(c),
        'source_corpus_sha256':corpus_sha,'packed_manifest_sha256':file_sha256(packing/'manifest.json'),
        'data_seed':23,'routing_seed':271828,'source_cursor':0,'source_windows':256,
        'model_seed':743,'numerical_weight_perturbation_std':.01,
        'recurrent_forward_steps':4,'recurrent_backward_cap':2,
        'native_ranks_per_job':32,'batch':1,'accumulation':8,'context':1024,
        'gradient_gate':.02,'nll_gate':.003,'cpu_deadline_unix':deadline,
        'optimizer_updates':0,'test_accessed':False,
        'scope':'CPU reference preparation for a subsequent bounded native gradient-only qualification. No permission to interrupt the active campaign or launch scientific training.'}
    write_state(out/'plan.json',plan)
    write_state(public/'preparation-plan.json',plan)
    records=[]
    try:
        packed=PackedCorpus.from_mapped(packing,1024,23,corpus_sha)
        counts=packed.counts(0,256)
        total=sum(counts)
        noise=np.lib.format.open_memmap(out/'initial-state.npy',mode='w+',dtype=np.float32,shape=(256,1024,c.dim))
        std=(2/(5*c.dim))**.5
        for index in range(256):
            generator=torch.Generator().manual_seed(1753+index)
            state=torch.empty(1024,c.dim)
            torch.nn.init.trunc_normal_(state,std=std,a=-3*std,b=3*std,generator=generator)
            noise[index]=state.numpy()
        noise.flush()
        noise_sha=file_sha256(out/'initial-state.npy')
        plan.update(packing_fingerprint=packed.fingerprint,real_targets=total,
            initial_state_file_sha256=noise_sha,domain_real_targets=dict(zip(['web','math','code'],counts)))
        write_state(out/'plan.json',plan)
        write_state(public/'preparation-plan.json',plan)
        for arm in ['ordinary','recurrent']:
            guard()
            recurrent=arm=='recurrent'
            begin=time.time()
            target=out/arm;target.mkdir()
            torch.manual_seed(743)
            config=reference_config(None,c,recurrent)
            if not recurrent:
                import parcae_lm.models.gpt.gpt as gpt
                gpt.LinearCrossEntropyLoss=PlainTiedHead
            model=config.construct_model().train()
            if sum(p.numel() for p in model.parameters())!=c.parameter_count(recurrent):
                raise AssertionError('Independent reference parameter inventory differs')
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(torch.randn_like(parameter)*.01)
            state=digest(model.state_dict())
            atomic_checkpoint(target/'qualification-state.pt',{'model':model.state_dict(),
                'config':asdict(c),'recurrent':recurrent,'qualification_only':True,
                'state_digest':state,'initial_state_sha256':noise_sha,'source_sha256':corpus_sha})
            checkpoint_sha=file_sha256(target/'qualification-state.pt')
            current_state=None
            if recurrent:
                model.initialize_state=lambda _:current_state
            def forward(batch,index):
                nonlocal current_state
                kwargs={}
                if recurrent:
                    current_state=torch.from_numpy(np.array(noise[index:index+1],copy=True))
                    kwargs['num_steps_pair']=torch.tensor([2,2])
                return model(batch.inputs,return_logits=True,**kwargs)['logits']
            def loss(logits,batch):
                return (F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none')
                        *batch.target_mask.flatten()).sum()
            index=routed_indices(0,0,0,32,1,8,271828)[0]
            batch=packed.batch([index],0)[0]
            with sdpa_kernel(SDPBackend.MATH):
                ref_logits=forward(batch,index)
                ref_loss=loss(ref_logits,batch)/batch.target_mask.sum()
                ref_loss.backward()
                gradients={name:p.grad.detach().clone() for name,p in model.named_parameters()}
                atomic_checkpoint(target/'local-cpu-fp32.pt',{'gradients':gradients,'loss':float(ref_loss.detach())})
                port=ParcaeCandidate(c,recurrent=recurrent,initialize=False).train()
                port.load_state_dict(model.state_dict())
                actual=port(batch.inputs,current_state,total_steps=4 if recurrent else 1,backward_cap=2 if recurrent else 1)
                port_loss=loss(actual,batch)/batch.target_mask.sum()
                port_loss.backward()
                error=norm=0.
                for name,parameter in port.named_parameters():
                    error+=float((parameter.grad-gradients[name]).double().square().sum())
                    norm+=float(gradients[name].double().square().sum())
                relative=(error/max(norm,1e-30))**.5
                logit_error=float((actual.detach()-ref_logits.detach()).abs().max())
                if relative>2e-5 or logit_error>2e-5:
                    raise AssertionError('Full1024-window port/reference mismatch')
                del port,actual,port_loss,ref_logits,ref_loss,gradients
                model.zero_grad(set_to_none=True);gc.collect()
                total_loss=0.
                for i in range(256):
                    guard()
                    batch=packed.batch([i],0)[0]
                    value=loss(forward(batch,i),batch)
                    (value/total).backward()
                    total_loss+=float(value.detach())
                    if (i+1)%16==0:
                        write_state(out/'heartbeat.json',{'unix':time.time(),'arm':arm,'cpu_windows':i+1})
                        print(json.dumps({'arm':arm,'cpu_windows':i+1}),flush=True)
            gradient={name:p.grad.detach().clone() for name,p in model.named_parameters()}
            if not all(torch.isfinite(g).all() for g in gradient.values()):
                raise AssertionError('Nonfinite independent CPU global gradient')
            atomic_checkpoint(target/'global-cpu-fp32.pt',{'gradients':gradient,'loss':total_loss/total})
            if digest(model.state_dict())!=state or file_sha256(target/'qualification-state.pt')!=checkpoint_sha:
                raise AssertionError('Gradient reference changed its qualification weights')
            row={'arm':arm,'parameters':c.parameter_count(recurrent),'real_targets':total,
                'local_port_reference_gradient_relative_l2':relative,'local_port_reference_logit_max_error':logit_error,
                'checkpoint_sha256':checkpoint_sha,'global_gradient_file_sha256':file_sha256(target/'global-cpu-fp32.pt'),
                'cpu_fp32_training_fixture_nll':total_loss/total,'active_seconds':time.time()-begin}
            records.append(row);write_state(public/'cpu-progress.json',{'records':records})
            del model,gradient;gc.collect()
        if any(file_sha256(p)!=h for p,h in plan['source_sha256'].items()):
            raise AssertionError('Reference preparation source changed')
        result={'status':'COMPLETE_INDEPENDENT_PARCAE_CPU_GLOBAL_REFERENCES','records':records,
            'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,
            'wall_seconds_including_imported_setup':time.time()-started,
            'scope':'Complete256-window natural-TRAIN FP32 reference gradients from pinned independent upstream code. Numerical-only perturbed states. No learning, native execution or optimizer qualification.'}
        write_state(out/'report.json',result);write_state(public/'cpu-report.json',result)
    except BaseException as error:
        result={'status':'STOPPED_PARCAE_CPU_REFERENCE_PREPARATION','error':str(error),'records':records,
            'assigned_chips':0,'optimizer_updates':0,'wall_seconds':time.time()-started,'test_accessed':False}
        write_state(out/'report.json',result);write_state(public/'cpu-report.json',result)
        raise


if __name__=='__main__':main()
