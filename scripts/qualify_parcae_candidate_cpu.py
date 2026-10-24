"""Whole-model CPU equivalence to pinned Parcae; no research data or training."""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA', '0')
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.flop_counter import FlopCounterMode

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig, matmul_ledger
from compute_trainium.queue import write_state


PIN = '69284c13746e849104f738d6d1a347b1f457df76'


def reference_config(upstream, c, recurrent):
    import parcae_lm
    kwargs = dict(n_embd=c.dim, intermediate_size=c.ffn_dim, num_attention_heads=c.heads,
        num_key_value_heads=c.heads, vocab_size=c.vocab_size, padded_vocab_size=c.vocab_size,
        padding_multiple=1, block_size=c.max_seq)
    if recurrent:
        kwargs.update(recurrent_embedding_dimension=c.dim,
            recurrent_intermediation_embedding_dimension=c.ffn_dim, recurrent_num_attention_heads=c.heads)
    return parcae_lm.create_config('parcae-small-140m' if recurrent else 'gpt-small-140m', **kwargs)


class PlainTiedHead(nn.Linear):
    """CPU constructor adapter for the GPT release's optional Triton head.

    The GPT readout below uses its existing tied-weight PyTorch forward path;
    no Triton or CUDA package is installed or altered.
    """
    def __init__(self, dim, vocab, **kwargs):
        if kwargs.get('transposed_weight'):
            raise ValueError('Only the published tied-head configuration is in scope')
        super().__init__(dim, vocab, bias=False)


def compare(c, recurrent, *, positions=16, batch=2, cases=None):
    torch.manual_seed(137)
    config = reference_config(None, c, recurrent)
    if not recurrent:
        import parcae_lm.models.gpt.gpt as gpt
        gpt.LinearCrossEntropyLoss = PlainTiedHead
    ref = config.construct_model().train()
    # Perturb away from the zero-output initialization so every internal map
    # participates in a meaningful gradient comparison.
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(torch.randn_like(p)*.01)
    port = ParcaeCandidate(c, recurrent=recurrent, initialize=False).train()
    port.load_state_dict(ref.state_dict(), strict=True)
    if sum(p.numel() for p in ref.parameters()) != c.parameter_count(recurrent):
        raise AssertionError('Unique reference/port parameters differ')
    generator = torch.Generator().manual_seed(811)
    tokens = torch.randint(c.vocab_size, (batch, positions), generator=generator)
    target = torch.randint(c.vocab_size, tokens.shape, generator=generator)
    mask = (torch.arange(batch*positions).reshape(batch, positions) % 5 != 0).float()
    state = ref.initialize_state(torch.empty(batch, positions, c.dim)) if recurrent else None
    if recurrent:
        ref.initialize_state = lambda _: state.clone()
    records = []
    cases = cases or ([(1, 1), (2, 2), (4, 2), (4, 4)] if recurrent else [(1, 1)])
    with sdpa_kernel(SDPBackend.MATH):
        for depth, cap in cases:
            ref.zero_grad(set_to_none=True)
            port.zero_grad(set_to_none=True)
            kwargs = {'num_steps_pair': torch.tensor([depth-min(depth, cap), min(depth, cap)])} if recurrent else {}
            upstream = ref(tokens, return_logits=True, **kwargs)['logits']
            with FlopCounterMode(display=False) as counter:
                result = port(tokens, state, total_steps=depth, backward_cap=cap)
                loss = (F.cross_entropy(result.flatten(0, 1), target.flatten(), reduction='none').reshape_as(mask)*mask).sum()/mask.sum()
                loss.backward()
            ref_loss = (F.cross_entropy(upstream.flatten(0, 1), target.flatten(), reduction='none').reshape_as(mask)*mask).sum()/mask.sum()
            ref_loss.backward()
            source = dict(ref.named_parameters())
            error = norm = maximum = 0.
            missing = []
            for name, p in port.named_parameters():
                expected = source[name].grad
                if p.grad is None or expected is None:
                    if (p.grad is None) != (expected is None):
                        missing.append(name)
                    continue
                delta = p.grad.detach()-expected
                error += float(delta.double().square().sum())
                norm += float(expected.double().square().sum())
                maximum = max(maximum, float(delta.abs().max()))
            relative = (error/max(norm, 1e-30))**.5
            output_error = float((result-upstream).abs().max().detach())
            expected_flops = matmul_ledger(c, positions, batch, recurrent=recurrent, total_steps=depth, backward_cap=cap)
            measured = counter.get_total_flops()
            if missing or output_error > 2e-5 or relative > 2e-5 or measured != expected_flops['training_matmul_flops']:
                raise AssertionError(json.dumps({'missing':missing, 'output_error':output_error,
                    'relative_gradient':relative, 'counted_flops':measured, 'ledger':expected_flops}))
            # Changing the future must not change any earlier prediction.
            changed = tokens.clone()
            changed[:, 9:] = (changed[:, 9:]+1) % c.vocab_size
            with torch.no_grad():
                future = port(changed, state, total_steps=depth, backward_cap=cap)
                prefix_error = float((future[:, :9]-result.detach()[:, :9]).abs().max())
            if prefix_error != 0:
                raise AssertionError('Candidate violates causal masking')
            records.append({'recurrent':recurrent, 'forward_steps':depth, 'backward_cap':cap,
                'model_dim':c.dim, 'vocab_size':c.vocab_size, 'context':positions, 'batch':batch,
                'maximum_logit_difference':output_error, 'relative_full_gradient_error':relative,
                'maximum_absolute_gradient_difference':maximum, 'causal_prefix_error':prefix_error,
                'unique_parameters':sum(p.numel() for p in port.parameters()),
                'counted_forward_backward_matmul_flops':measured, 'ledger':expected_flops})
    del ref, port
    gc.collect()
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--full-width', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    candidate_files = ['compute_trainium/parcae_candidate.py',
        'compute_trainium/stable_recurrence_candidate.py', 'scripts/qualify_parcae_candidate_cpu.py']
    candidate_hashes = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in candidate_files}
    actual = subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'], text=True).strip()
    if actual != PIN or dirty:
        raise ValueError('Clean pinned upstream checkout required')
    sys.path.insert(0, str(a.upstream))
    torch.set_num_threads(2)
    c = ParcaeCandidateConfig(dim=64, heads=2, ffn_dim=128, vocab_size=256, max_seq=32)
    records = compare(c, False)+compare(c, True)
    default = ParcaeCandidateConfig()
    if a.full_width:
        records += compare(default, False, positions=128, batch=1)
        records += compare(default, True, positions=128, batch=1, cases=[(4, 2)])
    meta = []
    for recurrent in [False, True]:
        with torch.device('meta'):
            model = ParcaeCandidate(default, recurrent=recurrent, initialize=False)
        meta.append({'recurrent':recurrent, 'parameters':sum(p.numel() for p in model.parameters()),
            'published_parameter_count':144323136 if recurrent else 143141184})
        if meta[-1]['parameters'] != meta[-1]['published_parameter_count']:
            raise AssertionError('Published140M inventory differs')
    report = {'status':'PASS_CPU_WHOLE_MODEL_REFERENCE_AND_EXECUTED_MATMUL_CHECKS',
        'upstream_commit':PIN,'records':records,'default_parameter_inventory':meta,
        'candidate_source_sha256':candidate_hashes,
        'source_sha256':{str(p.relative_to(a.upstream)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((a.upstream/'parcae_lm').rglob('*.py'))},
        'compatibility_adapter':'GPT optional Triton-head constructor replaced in-memory with a plain tied nn.Linear; its existing PyTorch tied output path is unchanged. No package/checkout modification.',
        'scope':'Numerical software fixtures, static explicit depths and identical supplied initial states/weights. No research data, native kernels, optimizer, sampled-depth DDP, recovery, throughput or learning qualification. Candidate initialization distributions do not reproduce upstream RNG draw order.',
        'prior_preparation_failure':'A first exploratory constructor probe disabled orthogonal initialization; the scaled-zero normal initializer divided by zero. Restoring the actual published orthogonal setting succeeds. It was not a training failure and has no exact elapsed receipt.',
        'optimizer_updates':0, 'assigned_chips':0, 'test_accessed':False,
        'wall_seconds_after_imports':time.time()-started}
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in candidate_hashes.items()):
        raise AssertionError('Candidate changed during qualification')
    write_state(a.output/'report.json', report)
    print(json.dumps({k:report[k] for k in ['status','records','default_parameter_inventory','wall_seconds_after_imports']},indent=2))


if __name__ == '__main__':
    main()
