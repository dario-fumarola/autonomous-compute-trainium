"""Pinned-upstream CPU parity for one prospective conventional control."""
import argparse
from dataclasses import asdict,replace
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.metrics import real_token_losses
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
from compute_trainium.parcae_training_candidate import gradient_comparison
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,atomic_checkpoint,file_sha256
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN,reference_config,PlainTiedHead

IDENTITY='ordinary8-cpu-parity-v2'
PUBLIC=Path('results/ordinary8_cpu_parity_20261026/v2')
DOC='docs/ORDINARY8_CPU_PARITY_20261026.md'
CALLS=dict(model_forward_calls_started=0,backward_calls_started=0,manual_model_updates_started=0)
CREATED=False


def ordinary_reference_config(config):
    # The convenience helper fixes only width/head sizes. Replacing n_layer
    # invokes GPTConfig.__post_init__,including its depth-dependent initializer.
    value=replace(reference_config(None,config,False),n_layer=config.layers)
    if (value.n_layer!=config.layers or value.n_embd!=config.dim
            or value.intermediate_size!=config.ffn_dim or value.num_attention_heads!=config.heads
            or value.num_key_value_heads!=config.heads or value.block_size!=config.max_seq
            or value.norm_eps!=config.norm_eps or value.rope_settings.rope_base!=config.rope_base
            or not value.tie_embeddings):
        raise AssertionError('Explicit complete ordinary reference geometry required')
    return value


def main():
    global CREATED
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/IDENTITY
    if out.exists():raise FileExistsError('Preserve the numerical preparation attempt')
    torch.set_num_threads(12);torch.set_num_interop_threads(1)
    def guard():
        require_inode_reserve([a.root,Path('/')])
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3 or min(shutil.disk_usage(a.root).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Protected numerical-preparation reserves failed')
        if time.time()-started>900:raise TimeoutError('Bounded CPU parity cap')
    guard();inventory_file=a.root/'ordinary-challenger-inventory-v1/report.json'
    inventory=json.loads(inventory_file.read_text())
    if inventory['status']!='COMPLETE_METADATA_ONLY_ORDINARY_CHALLENGER_INVENTORY':raise ValueError('Original geometric inventory required')
    for file,sha in inventory['source_sha256'].items():
        if file_sha256(file)!=sha:raise AssertionError('Original control inventory source changed')
    chosen=next(v for v in inventory['records'] if v['label']=='ordinary-8x640')
    c=ParcaeCandidateConfig(**chosen['config']);upstream=a.root/'upstream-parcae-69284c1'
    if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
            or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):
        raise ValueError('Clean pinned independent source required')
    sys.path.insert(0,str(upstream))
    parent_file=a.root/'fixed-two-ntp-main-v1/plan.json';parent=json.loads(parent_file.read_text())
    if file_sha256(parent_file)!=inventory['reference_plan_sha256']:raise AssertionError('Original scientific reference changed')
    files=[__file__,DOC,'scripts/audit_parcae_precise_training_point.py','scripts/qualify_parcae_candidate_cpu.py',
           'compute_trainium/parcae_candidate.py','compute_trainium/parcae_primitive_sdpa_candidate.py',
           'compute_trainium/parcae_fp16_candidate.py','compute_trainium/parcae_stages_candidate.py',
           'compute_trainium/parcae_training_candidate.py','compute_trainium/metrics.py','compute_trainium/training.py',
           'tests/test_ordinary8_reference.py']
    repo=Path(__file__).resolve().parents[1]
    sources={str(Path(f).resolve().relative_to(repo)):file_sha256(f) for f in files}
    out.mkdir();CREATED=True;PUBLIC.mkdir(parents=True)
    write_state(out/'registration.json',dict(source_sha256=sources,inventory_sha256=file_sha256(inventory_file),
        model=asdict(c),qualification_only=True,seed=1009,windows=[0,1,2,3],seconds_cap=900,
        gates=dict(maximum_ntp_difference=.00002,maximum_logit_absolute_error=.0002,gradient_relative_l2=.002),
        native_launch_allowed=False,scientific_fit_admitted=False,scope=Path(DOC).read_text()))
    import parcae_lm.models.gpt.gpt as gpt
    gpt.LinearCrossEntropyLoss=PlainTiedHead
    reference_configured=ordinary_reference_config(c)
    torch.manual_seed(1009);reference=reference_configured.construct_model().train()
    if sum(p.numel() for p in reference.parameters())!=chosen['unique_parameters']:
        raise AssertionError('Pinned upstream parameter inventory differs')
    initial_digest=digest(dict(sorted(reference.state_dict().items())));initial=out/'initial-state-ordinary.pt'
    atomic_checkpoint(initial,dict(model=reference.state_dict(),config=asdict(c),qualification_only=True,
        model_seed=1009,recurrent=False,model_digest=initial_digest,upstream_commit=PIN,
        purpose='ordinary8_cpu_qualification_initializer',optimizer_updates=0))
    initial_sha=file_sha256(initial);saved=torch.load(initial,map_location='cpu',mmap=True,weights_only=False)
    if digest(dict(sorted(saved['model'].items())))!=initial_digest:raise AssertionError('Original initializer readback differs')
    local=ParcaePrimitiveSDPACandidate(c,recurrent=False,initialize=False,fp16_body=False)
    local.load_state_dict(saved['model'],strict=True);del saved
    if digest(dict(sorted(local.state_dict().items())))!=initial_digest:raise AssertionError('Original local tensors differ')
    packed=PackedCorpus.from_mapped(a.root/parent['data']['packed'],1024,23,parent['data']['corpus_sha256'])
    if packed.fingerprint!=parent['data']['packing_fingerprint']:raise AssertionError('Original authentic TRAIN packing changed')
    windows=[packed.batch([i],0)[0] for i in range(4)];targets=sum(int(v.target_mask.sum()) for v in windows)
    records=[]
    for phase in ['original','activated_projection_probe']:
        expected_sum=actual_sum=maximum=0.;reference.zero_grad(set_to_none=True);local.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH):
            for item in windows:
                guard();CALLS['model_forward_calls_started']+=1
                expected=reference(item.inputs,return_logits=True)['logits'];CALLS['model_forward_calls_started']+=1
                actual=local(item.inputs,None,total_steps=1,backward_cap=1)
                maximum=max(maximum,float((expected.detach()-actual.detach()).abs().max()))
                left=(F.cross_entropy(expected.flatten(0,1),item.targets.flatten(),reduction='none')*item.target_mask.flatten()).sum()
                right=real_token_losses(actual,item.targets,item.target_mask).sum()
                expected_sum+=float(left.detach());actual_sum+=float(right.detach())
                CALLS['backward_calls_started']+=1;(left/targets).backward()
                CALLS['backward_calls_started']+=1;(right/targets).backward()
                del expected,actual,left,right
        gold={n:p.grad.detach() for n,p in reference.named_parameters()};observed={n:p.grad.detach() for n,p in local.named_parameters()}
        delta=gradient_comparison(gold,observed);difference=(actual_sum-expected_sum)/targets
        passed=(maximum<=.0002 and abs(difference)<=.00002 and delta['passes_two_percent_gate']
                and delta['overall']['relative_l2']<=.002
                and all(v['relative_l2']<=.002 for v in delta['families'].values()))
        records.append(dict(phase=phase,real_targets=targets,upstream_ntp_nll=expected_sum/targets,
                            local_ntp_nll=actual_sum/targets,ntp_difference=difference,
                            maximum_logit_absolute_error=maximum,gradient=delta,passes_gate=passed))
        write_state(out/'progress.json',dict(records=records))
        if not passed:raise AssertionError('Registered CPU model-parity gate failed')
        if phase=='original':
            # Same observed natural gradient;this is a numerical state probe,
            # not a qualification of an optimizer or a scientific update.
            other=dict(local.named_parameters())
            CALLS['manual_model_updates_started']+=2
            with torch.no_grad():
                for name,parameter in reference.named_parameters():
                    parameter.add_(gold[name],alpha=-.01);other[name].add_(gold[name],alpha=-.01)
            if digest(dict(sorted(local.state_dict().items())))!=digest(dict(sorted(reference.state_dict().items()))):
                raise AssertionError('Identical numerical state intervention differs')
    guard()
    if file_sha256(initial)!=initial_sha:raise AssertionError('Qualification probe changed saved original weights')
    result=dict(status='PASS_ORDINARY8_CPU_MODEL_PARITY',source_sha256=sources,records=records,
        model=asdict(c),upstream_n_layer=reference_configured.n_layer,
        parameters=chosen['unique_parameters'],initial_model_digest=initial_digest,
        initializer_sha256=initial_sha,upstream_commit=PIN,model_forwards=16,backward_calls=16,
        manual_qualification_model_updates=2,target_optimizer_steps=0,research_optimizer_updates=0,
        assigned_chips=0,test_accessed=False,terminal_alpha_spent=0,native_launch_allowed=False,
        scientific_fit_admitted=False,wall_seconds=time.time()-started,
        scope='Two CPU FP32 implementations,on four authentic TRAIN windows before/after the same natural-gradient state intervention. Updated probe states are not saved or adopted. No native FP16/full-global/optimizer/recovery/throughput/learning claim.')
    write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
    print(json.dumps(dict(status=result['status'],parameters=result['parameters'],wall_seconds=result['wall_seconds'])))


if __name__=='__main__':
    process_start=time.time()
    try:main()
    except BaseException as error:
        if CREATED:
            parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--root',type=Path)
            args,_=parser.parse_known_args()
            failure=dict(status='STOPPED_ORDINARY8_CPU_PREPARATION',error=str(error),calls_started=CALLS,
                assigned_chips=0,research_optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0,
                wall_seconds=time.time()-process_start,
                scope='Attempted-call counts include an interrupted call;no qualification state is adopted for science.')
            if args.root:write_state(args.root/IDENTITY/'failure.json',failure)
            write_state(PUBLIC/'failure.json',failure)
        raise
