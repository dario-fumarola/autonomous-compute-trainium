import copy
import pytest
import torch
from torch.nn import functional as F

from compute_trainium.model import ModelConfig
from compute_trainium.budgeted_interleaved_factory import make_model, graph_spec, GRAPH_SPECS, matrix_work
from compute_trainium.budgeted_interleaved_reference import matrix_work as reference_work
from scripts.run_budgeted_interleaved_qualification import timing_summary


@pytest.mark.parametrize('selector', list(GRAPH_SPECS))
def test_qualified_buffer_graph_matches_reference_full_gradients(selector):
    torch.manual_seed(40341)
    c = ModelConfig(dim=16, layers=2, heads=2, kv_heads=1, ffn_dim=32, vocab_size=23, max_seq=4)
    p = dict(model_config=c.__dict__, workspace_mechanism='budgeted_persistent_generated_state')
    a = make_model(p).double()
    b = make_model(p, explicit_attention=True).double()
    b.load_state_dict(a.state_dict())
    x, y = torch.randint(0, 23, (2, 4)), torch.randint(0, 23, (2, 4))
    aa, bb = a(x, selector, 'latent'), b(x, selector, 'latent')
    torch.testing.assert_close(aa, bb, atol=1e-11, rtol=1e-10)
    for output in (aa, bb):
        F.cross_entropy(output.flatten(0, 1), y.flatten()).backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-9)
    budget, rounds = GRAPH_SPECS[selector]
    assert matrix_work(c, 4, selector, batch=2) == reference_work(c, 4, budget, rounds, 2)


@pytest.mark.parametrize('selector,mode', [(True, 'latent'), (2, 'latent'), (12, 'pause'), (25., 'latent')])
def test_unknown_compatibility_identifiers_rejected(selector, mode):
    with pytest.raises(ValueError):
        graph_spec(selector, mode)


def test_timing_excludes_each_first_use_without_dropping_slow_repeats():
    ids = list(GRAPH_SPECS)
    seen = set(); rows = []
    for update in range(1, 105):
        key = 25 if update == 104 else ids[(update-1) % len(ids)]
        rows.append(dict(update=update, extra_steps=key, seconds=.2 if key in seen else 100.))
        seen.add(key)
    summary = timing_summary(rows)
    assert summary['throughput_pass']
    assert summary['remaining_updates'] == 99
    assert len(summary['first_graph_uses']) == 5
    changed = copy.deepcopy(rows)
    changed[-1]['seconds'] = 20.
    assert not timing_summary(changed)['throughput_pass']
    with pytest.raises(AssertionError):
        timing_summary(rows[:-1])
