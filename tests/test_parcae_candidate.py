import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.flop_counter import FlopCounterMode

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig, matmul_ledger


def test_published_inventory_and_tied_head():
    c = ParcaeCandidateConfig()
    with torch.device('meta'):
        normal = ParcaeCandidate(c, recurrent=False, initialize=False)
        recurrent = ParcaeCandidate(c, recurrent=True, initialize=False)
    assert sum(p.numel() for p in normal.parameters()) == 143141184
    assert sum(p.numel() for p in recurrent.parameters()) == 144323136
    assert recurrent.lm_head.weight is recurrent.transformer.wte.weight


def test_forward_truncation_and_causality():
    torch.manual_seed(313)
    c = ParcaeCandidateConfig(dim=32, heads=2, ffn_dim=64, vocab_size=64, max_seq=16)
    model = ParcaeCandidate(c, initialize=False)
    tokens = torch.arange(8)[None]
    state = torch.randn(1, 8, 32)
    full = model(tokens, state, total_steps=3, backward_cap=3)
    tail = model(tokens, state, total_steps=3, backward_cap=1)
    torch.testing.assert_close(full, tail, rtol=0, atol=0)
    changed = tokens.clone()
    changed[:, 5:] += 8
    other = model(changed, state, total_steps=3, backward_cap=1)
    torch.testing.assert_close(full[:, :5], other[:, :5], rtol=0, atol=0)


def test_execution_counter_matches_ledger_with_head_and_truncation():
    torch.manual_seed(23)
    c = ParcaeCandidateConfig(dim=32, heads=2, ffn_dim=64, vocab_size=64, max_seq=16)
    model = ParcaeCandidate(c, initialize=False)
    tokens, state = torch.arange(8)[None], torch.randn(1, 8, 32)
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        model(tokens, state, total_steps=4, backward_cap=1).square().mean().backward()
    ledger = matmul_ledger(c, 8, total_steps=4, backward_cap=1)
    assert counter.get_total_flops() == ledger['training_matmul_flops']
    assert ledger['head_forward_matmul_flops'] == 2*8*32*64
