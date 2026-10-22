import copy
import math

import pytest
import torch

from compute_trainium.data import (content_hash, document_windows, dummy_pause_mask,
                                   layout, prediction_spans, source_group, split_for_group)
from compute_trainium.metrics import (DocumentScore, paired_group_interval, paired_three_seed_interval,
                                      real_token_losses, summarize)
from compute_trainium.model import LanguageModel, ModelConfig


def tiny():
    torch.manual_seed(19)
    return LanguageModel(ModelConfig(dim=16, layers=2, heads=4, kv_heads=2,
                                    ffn_dim=32, vocab_size=37, max_seq=128))


def test_windows_cover_every_target_once_and_exclude_boundaries():
    tokens = list(range(4, 31))
    recovered = []
    for source, mask in document_windows(tokens, 8):
        recovered.extend(source[1:][mask].tolist())
        assert source.numel() == 9
    assert recovered == tokens


def test_pause_alignment_preserves_targets_and_scores_last_pause():
    source, mask = next(document_windows(list(range(4, 16)), 12))
    b = layout(source[None], mask[None], pauses=3, interval=4)
    assert b.targets[0][b.target_mask[0]].tolist() == list(range(4, 16))
    assert b.prediction_positions.tolist() == [0, 1, 2, 6, 7, 8, 9, 13, 14, 15, 16, 20]
    assert b.inputs[0, b.prediction_positions[3]].item() == 0
    assert b.pause_mask.sum().item() == 9


def test_causality_and_gradients_through_pause_embedding():
    model = tiny()
    source = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10, 11]])
    b = layout(source, torch.ones(1, 8, dtype=torch.bool), pauses=2, interval=4)
    logits = model(b.inputs, b.prediction_positions)
    changed = b.inputs.clone()
    changed[0, -1] = 35
    assert torch.equal(logits[:, :-1], model(changed, b.prediction_positions)[:, :-1])
    real_token_losses(logits, b.targets, b.target_mask).mean().backward()
    assert model.embedding.weight.grad[0].abs().sum() > 0


def test_dummy_pause_hidden_states_cannot_receive_source_information():
    model = tiny()
    b = layout(torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10, 11]]),
               torch.ones(1, 8, dtype=torch.bool), pauses=2, interval=4)
    mask = dummy_pause_mask(b.pause_mask)
    changed = b.inputs.clone()
    changed[:, ~b.pause_mask] = 30
    a, z = model.hidden(b.inputs, mask), model.hidden(changed, mask)
    torch.testing.assert_close(a[:, b.pause_mask], z[:, b.pause_mask], rtol=0, atol=0)
    assert not torch.equal(a[:, ~b.pause_mask], z[:, ~b.pause_mask])


def test_static_spans_equal_gather_in_logits_and_full_parameter_gradients():
    a, b = tiny(), tiny()
    batch = layout(torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10, 11]]),
                   torch.ones(1, 8, dtype=torch.bool), pauses=2, interval=4)
    x = a(batch.inputs, batch.prediction_positions)
    y = b(batch.inputs, prediction_spans=prediction_spans(batch.prediction_positions))
    torch.testing.assert_close(x, y, atol=0, rtol=0)
    x.square().sum().backward()
    y.square().sum().backward()
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_causal_tail_padding_preserves_every_real_prediction_and_full_gradient():
    from compute_trainium.data import pad_layout
    a, b = tiny(), tiny()
    batch = layout(torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10, 11]]),
                   torch.ones(1, 8, dtype=torch.bool), pauses=2, interval=4)
    padded = pad_layout(batch, 32)
    x = a(batch.inputs, batch.prediction_positions)
    y = b(padded.inputs, padded.prediction_positions)
    torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-5)
    x.square().sum().backward()
    y.square().sum().backward()
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
    assert torch.equal(batch.targets, padded.targets)
    assert torch.equal(batch.target_mask, padded.target_mask)


def test_nll_uses_real_target_denominator_and_fp32_ce():
    logits = torch.zeros(1, 3, 5, requires_grad=True)
    loss = real_token_losses(logits, torch.tensor([[1, 2, 3]]), torch.tensor([[True, False, True]]))
    assert loss.sum().item() / 2 == pytest.approx(math.log(5))
    loss.sum().backward()
    assert torch.equal(logits.grad[:, 1], torch.zeros(1, 5))


def test_explicit_nll_matches_cross_entropy_and_gradients():
    torch.manual_seed(41)
    a = torch.randn(2, 5, 37, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    y = torch.randint(37, (2, 5))
    mask = torch.tensor([[True, False, True, True, False]]).expand(2, -1)
    loss = real_token_losses(a, y, mask).sum()
    reference = (torch.nn.functional.cross_entropy(b.flatten(0, 1), y.flatten(),
                                                   reduction="none").view(2, 5) * mask).sum()
    torch.testing.assert_close(loss, reference)
    loss.backward()
    reference.backward()
    torch.testing.assert_close(a.grad, b.grad)


def test_global_source_split_and_normalized_duplicate_key():
    assert source_group("web", "https://www.example.org/a") == source_group("math", "http://example.org/b")
    assert split_for_group(source_group("code", "Owner/Repo")) == split_for_group("repo:owner/repo")
    assert content_hash("hello\n world") == content_hash("hello world")
    with pytest.raises(ValueError):
        source_group("math", "missing")


def test_domain_weighting_and_paired_cluster_statistics():
    a = [DocumentScore(domain, f"{domain}/{i}", f"g{i}", 10*(i+1), i+1)
         for domain in ("web", "math", "code") for i in range(8)]
    b = [DocumentScore(x.domain, x.document, x.group, x.nll_sum-.2*x.tokens, x.tokens) for x in a]
    assert summarize(a)["weighted_nll"] == pytest.approx(10)
    result = paired_group_interval(a, b, draws=200)
    assert result["gain_nats"] == pytest.approx(.2)
    assert result["interval_95"] == pytest.approx([.2, .2])
    with pytest.raises(ValueError):
        paired_group_interval(a, b[:-1])


def test_exact_parameter_count_and_lm_head_cost_when_tied():
    m = tiny()
    assert m.config.parameter_count() == sum(p.numel() for p in m.parameters())
    c = m.config.flops(64, 72)
    assert c["lm_head_forward_flops"] == 2*64*16*37
    assert c["training_matmul_flops"] > m.config.flops(64, 64)["training_matmul_flops"]


def test_seed_variance_is_distinct_from_document_uncertainty():
    r = paired_three_seed_interval({23: -.01, 47: .01, 71: .03})
    assert r["mean_gain_nats"] == pytest.approx(.01)
    assert r["interval_95"][0] < 0 < r["interval_95"][1]
    with pytest.raises(ValueError):
        paired_three_seed_interval({23: .01})


def test_end_to_end_eval_scores_each_real_token_with_pause_padding():
    from compute_trainium.evaluate import evaluate
    class Uniform(torch.nn.Module):
        def forward(self, tokens, prediction_positions):
            return torch.zeros(tokens.shape[0], prediction_positions.numel(), 37)
    docs = [{"domain": d, "id": d, "group": d, "split": "dev", "tokens": [4, 7, 9]*89}
            for d in ("web", "math", "code")]
    scores = evaluate(Uniform(), docs, context=128, pauses=16)
    assert [s.tokens for s in scores] == [267]*3
    assert summarize(scores)["weighted_nll"] == pytest.approx(math.log(37))


def test_optimizer_resume_survives_late_pause_switch():
    m = tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    source = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10, 11]])
    def step(pauses):
        b = layout(source, torch.ones(1, 8, dtype=torch.bool), pauses=pauses, interval=4)
        opt.zero_grad(set_to_none=True)
        loss = real_token_losses(m(b.inputs, b.prediction_positions), b.targets, b.target_mask).mean()
        loss.backward()
        opt.step()
    step(0)
    weights, state = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
    step(2)
    expected, moments = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
    m.load_state_dict(weights)
    opt.load_state_dict(state)
    step(2)
    for k in expected:
        assert torch.equal(m.state_dict()[k], expected[k])
    for i in moments["state"]:
        for k, value in moments["state"][i].items():
            assert torch.equal(opt.state_dict()["state"][i][k], value)
