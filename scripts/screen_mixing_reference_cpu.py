"""Numerical fixtures comparing local control with independently pinned blocks."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.mixing_control_reference import construct_reference
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_mixing_control import MixingOnlyFutureSpanCandidate
from compute_trainium.parcae_training_candidate import gradient_comparison
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.qualify_parcae_candidate_cpu import PIN


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();begin=time.time();torch.set_num_threads(2)
    upstream=args.root/'upstream-parcae-69284c1'
    if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
            or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):
        raise AssertionError('Clean pinned independent blocks required')
    sys.path.insert(0,str(upstream))
    torch.manual_seed(2831)
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=41,max_seq=16)
    span=FutureSpanConfig(context=16,horizon=4,stride=2)
    local=MixingOnlyFutureSpanCandidate(config,span,initialize=False,fp16_body=False)
    with torch.no_grad():
        for parameter in local.parameters():parameter.add_(torch.randn_like(parameter)*.01)
    reference=construct_reference(config,local.state_dict())
    tokens=torch.arange(16)[None]%41;noise=torch.randn(1,16,32)
    reference.initialize_state=lambda _:noise
    hidden={};hook=reference.transformer.ln_f.register_forward_hook(lambda m,ins,out:hidden.update(value=out))
    rows=[]
    with sdpa_kernel(SDPBackend.MATH):
        for depth in [1,2,3,5]:
            if time.time()-begin>60:raise TimeoutError('CPU fixture cap')
            local.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
            prediction,aux=local(tokens,noise,total_steps=depth,backward_cap=depth)
            expected=reference(tokens,return_logits=True,num_steps_pair=torch.tensor([0,depth]))['logits']
            expected_aux=reference.future_lm_head(hidden['value'].index_select(1,reference.future_positions))
            torch.testing.assert_close(prediction,expected,atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(aux,expected_aux,atol=1e-5,rtol=1e-5)
            (prediction.square().mean()+.1*aux.square().mean()).backward()
            (expected.square().mean()+.1*expected_aux.square().mean()).backward()
            actual={n:p.grad for n,p in local.named_parameters()}
            gold={n:p.grad for n,p in reference.named_parameters()}
            if set(actual)!=set(gold) or any(g is None for g in [*actual.values(),*gold.values()]):
                raise AssertionError('Every parameter gradient required')
            comparison=gradient_comparison(gold,actual)
            if comparison['overall']['relative_l2']>1e-5:
                raise AssertionError('Independent control gradient differs')
            rows.append(dict(mixing_count=depth,transformer_core_executions=1,
                maximum_logit_error=float((prediction-expected).detach().abs().max()),gradient=comparison))
    hook.remove()
    result=dict(status='PASS_INDEPENDENT_CPU_MIXING_CONTROL_FIXTURES',records=rows,
        upstream_commit=PIN,wall_seconds=time.time()-begin,assigned_chips=0,research_optimizer_updates=0,
        research_examples=0,test_accessed=False,terminal_alpha_spent=0,
        source_sha256={p:file_sha256(p) for p in ['scripts/screen_mixing_reference_cpu.py',
            'compute_trainium/mixing_control_reference.py','compute_trainium/parcae_mixing_control.py']},
        scope='Numerical software fixtures only,not a research corpus or actual-size/native qualification. Nonzero perturbed weights;independent pinned upstream blocks and adapter;complete parameter gradients.')
    out=args.root/'small-core-mixing-reference-cpu-v1';out.mkdir(exist_ok=False)
    public=Path('results/small_core_mixing_control_preparation_20261025')
    write_state(out/'report.json',result);write_state(public/'independent-reference.json',result)
    print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}))


if __name__=='__main__':main()
