"""Small random arrays are software fixtures, never research examples."""
import pytest
import torch
from torch.nn import functional as F

from compute_trainium.ephemeral_workspace import (
    EphemeralWorkspace, ephemeral_allowed, matrix_work, reused_source_matrix_work)
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.model import ModelConfig


def setup(explicit=False):
    torch.manual_seed(40271)
    c = ModelConfig(dim=16, layers=2, heads=2, kv_heads=1,
                    ffn_dim=32, vocab_size=23, max_seq=5)
    m = EphemeralWorkspace(c, explicit_attention=explicit).double()
    return m, torch.randint(0, 23, (2, 5)), torch.randint(0, 23, (2, 5))


def test_mask_has_only_source_prefix_and_own_latent():
    n = 5
    allowed = ephemeral_allowed(n)
    for q in range(2*n):
        for k in range(2*n):
            expected = k <= q % n if k < n else q >= n and k == q
            assert allowed[q, k].item() == expected


@pytest.mark.parametrize('explicit', [False, True])
def test_parallel_matches_independent_prefix_outputs_and_all_gradients(explicit):
    m, x, y = setup(explicit)
    a, z = m(x, return_states=True)
    b, zs = m.sequential_reference(x, return_states=True)
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(z, zs, atol=1e-11, rtol=1e-10)
    F.cross_entropy(a.flatten(0, 1), y.flatten()).backward()
    expected = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    F.cross_entropy(b.flatten(0, 1), y.flatten()).backward()
    for n, p in m.named_parameters():
        torch.testing.assert_close(expected[n], p.grad, atol=1e-10, rtol=1e-9)


def test_no_other_latent_influence_but_current_state_and_source_gradients_exist():
    m, x, y = setup(True)
    source = m.embeddings(x)
    source.retain_grad()
    z = m.body_at(source, torch.arange(x.shape[1]))
    z.retain_grad()
    h = m.local_body(source, z)
    F.cross_entropy(F.linear(h[:, -1], m.prediction_weight), y[:, -1]).backward()
    assert torch.count_nonzero(z.grad[:, :-1]) == 0
    assert z.grad[:, -1].norm() > 1e-10
    assert source.grad[:, :-1].norm() > 1e-10
    changed = z.detach().clone()
    changed[:, :-1] += 10
    torch.testing.assert_close(m.local_body(source, z)[:, -1],
                               m.local_body(source, changed)[:, -1], atol=0, rtol=0)


def test_future_invariance_and_same_initial_parameters():
    m, x, _ = setup()
    persistent = InterleavedWorkspace(m.config).double()
    persistent.load_state_dict(m.state_dict(), strict=True)
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in persistent.parameters())
    changed = x.clone()
    changed[:, 3:] = (changed[:, 3:] + 1) % m.config.vocab_size
    torch.testing.assert_close(m(x)[:, :3], m(changed)[:, :3], atol=0, rtol=0)
    # At the first position no earlier latent exists in either mechanism.
    torch.testing.assert_close(m(x)[:, :1], persistent.sequential(x[:, :1]), atol=1e-11, rtol=1e-10)


def test_sdpa_explicit_full_gradients_and_optimizer_step_agree():
    a, x, y = setup()
    b, _, _ = setup(True)
    b.load_state_dict(a.state_dict())
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, foreach=False) for m in (a, b)]
    for m, opt in zip((a, b), optimizers):
        opt.zero_grad()
        F.cross_entropy(m(x).flatten(0, 1), y.flatten()).backward()
        opt.step()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(p, q, atol=1e-11, rtol=1e-9)
    for s, t in zip(optimizers[0].state.values(), optimizers[1].state.values()):
        for key in s:
            torch.testing.assert_close(s[key], t[key], atol=1e-11, rtol=1e-9)


@pytest.mark.parametrize('reuse', [False, True])
def test_full_backward_work_ledger_counts_masked_attention(reuse):
    from torch.utils._python_dispatch import TorchDispatchMode
    class Counter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.flops = 0
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.mm.default:
                a, b = args
                self.flops += 2*a.shape[0]*a.shape[1]*b.shape[1]
            elif func is torch.ops.aten.bmm.default:
                a, b = args
                self.flops += 2*a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]
            return func(*args, **(kwargs or {}))
    m, x, y = setup(True)
    counter = Counter()
    with counter:
        logits = m.reused_source(x) if reuse else m(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        forward = counter.flops
        loss.backward()
    ledger = reused_source_matrix_work if reuse else matrix_work
    expected = ledger(m.config, 5, batch=2)
    assert forward == expected['forward_matmul_flops']
    assert counter.flops == expected['training_matmul_flops']


@pytest.mark.parametrize('explicit', [False, True])
def test_reused_source_matches_outputs_full_gradients_and_step(explicit):
    a, x, y = setup(explicit)
    b, _, _ = setup(explicit)
    b.load_state_dict(a.state_dict())
    outputs = (a(x), b.reused_source(x))
    torch.testing.assert_close(*outputs, atol=1e-11, rtol=1e-10)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, foreach=False) for m in (a, b)]
    for output, opt in zip(outputs, optimizers):
        opt.zero_grad()
        F.cross_entropy(output.flatten(0, 1), y.flatten()).backward()
        opt.step()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(p, q, atol=1e-11, rtol=1e-9)
    for s, t in zip(optimizers[0].state.values(), optimizers[1].state.values()):
        for key in s:
            torch.testing.assert_close(s[key], t[key], atol=1e-11, rtol=1e-9)


@pytest.mark.parametrize('length', [0, -1, True, 1.5])
def test_invalid_lengths(length):
    with pytest.raises(ValueError):
        ephemeral_allowed(length)


def test_explicit_factory_and_exact_ledger_adapter():
    from compute_trainium.ephemeral_model_factory import make_model, matrix_work as adapter_work
    m, x, _ = setup()
    plan = dict(model_config=m.config.__dict__, workspace_mechanism='ephemeral_generated_state',
                ephemeral_training_graph='reused_source_kv')
    cpu = make_model(plan)
    native = make_model(plan, explicit_attention=True)
    cpu.double().load_state_dict(m.state_dict())
    native.double().load_state_dict(m.state_dict())
    torch.testing.assert_close(cpu(x, 0, 'latent'), native(x, 0, 'latent'), atol=1e-11, rtol=1e-10)
    assert adapter_work(m.config, 5, 0, batch=2) == reused_source_matrix_work(m.config, 5, batch=2)
    for bad in ({}, {**plan, 'workspace_mechanism': 'persistent'}):
        with pytest.raises(ValueError):
            make_model(bad)
    for rounds, mode in [(1, 'latent'), (True, 'latent'), (0, 'ordinary')]:
        with pytest.raises(ValueError):
            cpu(x, rounds, mode)
        with pytest.raises(ValueError):
            adapter_work(m.config, 5, rounds, mode)


def test_qualification_waits_for_every_completed_audit(tmp_path):
    import json
    from scripts.run_ephemeral_qualification import completed_replication_fits
    for seed in (730201, 730237):
        for role in ('latent', 'ordinary', 'pause'):
            out = tmp_path/f'interleaved-seed-replication-v1-s{seed}-{role}'
            out.mkdir()
            (out/'report.json').write_text(json.dumps(dict(status='COMPLETE_AUDITED_REPLICATION_FIT',
                audit=dict(status='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS'))))
    assert len(completed_replication_fits(tmp_path)) == 6
    last = tmp_path/'interleaved-seed-replication-v1-s730237-pause/report.json'
    last.write_text(json.dumps(dict(status='COMPLETE_AUDITED_REPLICATION_FIT', audit=dict(status='FAIL'))))
    with pytest.raises(AssertionError):
        completed_replication_fits(tmp_path)
    last.unlink()
    with pytest.raises(FileNotFoundError):
        completed_replication_fits(tmp_path)
