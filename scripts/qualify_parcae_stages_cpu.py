"""Static-stage AOT autograd equivalence; no native or scientific training."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig
from compute_trainium.parcae_stages_candidate import ParcaeStagesCandidate
from compute_trainium.queue import write_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.set_num_threads(2)
    files = ['compute_trainium/parcae_candidate.py','compute_trainium/parcae_stages_candidate.py',
             'compute_trainium/stable_recurrence_candidate.py','scripts/qualify_parcae_stages_cpu.py']
    hashes = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    c = ParcaeCandidateConfig(dim=64, heads=2, ffn_dim=128, vocab_size=256, max_seq=32)
    records = []
    for recurrent in [False, True]:
        torch.manual_seed(271)
        ref = ParcaeCandidate(c,recurrent=recurrent,initialize=False).to('cpu')
        staged = ParcaeStagesCandidate(c,recurrent=recurrent,initialize=False).to('cpu')
        staged.load_state_dict(ref.state_dict())
        before_names = list(staged.state_dict())
        staged.compile_stages('aot_eager')
        if list(staged.state_dict()) != before_names:
            raise AssertionError('Compiling stages changed checkpoint names')
        x = torch.arange(32).reshape(2, 16)
        state = torch.randn(2, 16, c.dim) if recurrent else None
        cases = [(1,1),(2,2),(4,2),(3,1)] if recurrent else [(1,1)]
        with sdpa_kernel(SDPBackend.MATH):
            for total, cap in cases:
                ref.zero_grad(set_to_none=True)
                staged.zero_grad(set_to_none=True)
                expected = ref(x,state,total_steps=total,backward_cap=cap)
                actual = staged(x,state,total_steps=total,backward_cap=cap)
                expected.square().mean().backward()
                actual.square().mean().backward()
                output_error = float((actual.detach()-expected.detach()).abs().max())
                difference = norm = 0.
                for a,b in zip(staged.parameters(),ref.parameters(),strict=True):
                    if (a.grad is None) != (b.grad is None):
                        raise AssertionError('Stage boundary dropped a parameter gradient')
                    if a.grad is not None:
                        difference += float((a.grad-b.grad).double().square().sum())
                        norm += float(b.grad.double().square().sum())
                relative = (difference/max(norm,1e-30))**.5
                if output_error > 2e-5 or relative > 2e-5:
                    raise AssertionError('AOT-stage numerical mismatch')
                records.append({'recurrent':recurrent,'forward_steps':total,'backward_cap':cap,
                    'maximum_logit_difference':output_error,'full_gradient_relative_l2':relative,
                    'checkpoint_parameter_names_unchanged':True})
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise AssertionError('Stage source changed during qualification')
    report = {'status':'PASS_CPU_FULLGRAPH_STATIC_STAGES_AND_AUTOGRAD','records':records,
        'source_sha256':hashes,'assigned_chips':0,'test_accessed':False,'optimizer_updates':0,
        'scope':'AOT-eager CPU compiler with fullgraph=True/dynamic=False on each fixed-shape stage. Host loop counts vary without compiling a different whole unroll. Same model function/parameters and tail truncation. No native compilation, per-sequence DDP, throughput or model quality claim.',
        'wall_seconds_after_imports':time.time()-started}
    write_state(args.output/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
