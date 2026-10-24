import copy

import pytest
import torch

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_candidate import ParcaeOptimizerCandidate
from compute_trainium.parcae_training_candidate import (
    configure_optimizer, finish_gradients, initial_state, poisson_tail, validate_resume,
    map_after_shared_validation,
)
from compute_trainium.training import PackedCorpus, file_sha256


def test_stateless_state_and_depth_replay_without_global_rng_effect():
    config = ParcaeCandidateConfig(dim=32, heads=2, ffn_dim=64, vocab_size=128)
    before = torch.get_rng_state()
    a = initial_state(config, 47, 2009)
    b = initial_state(config, 47, 2009)
    assert torch.equal(a, b)
    assert not torch.equal(a, initial_state(config, 48, 2009))
    assert torch.equal(before, torch.get_rng_state())
    values = [poisson_tail(8, 4, step, rank) for step in range(32) for rank in range(4)]
    assert values == [poisson_tail(8, 4, step, rank) for step in range(32) for rank in range(4)]
    assert all(n >= 1 and k == min(n, 4) for n,k in values)
    assert len(set(values)) > 3


def test_global_normalization_with_heterogeneous_depths_and_masks():
    torch.manual_seed(919)
    c = ParcaeCandidateConfig(dim=32, heads=2, ffn_dim=64, vocab_size=128, max_seq=8)
    expected = ParcaeCandidate(c, initialize=False)
    replicas = [copy.deepcopy(expected) for _ in range(2)]
    tokens = [torch.randint(128, (1,8)) for _ in range(4)]
    targets = [torch.randint(128, (1,8)) for _ in range(4)]
    masks = [torch.arange(8) < (3+i) for i in range(4)]
    states = [torch.randn(1,8,32) for _ in range(4)]
    count = sum(int(m.sum()) for m in masks)
    for index in range(4):
        n, cap = [(1,1),(3,2),(2,2),(4,2)][index]
        for model, factor in [(expected,1), (replicas[index%2],2)]:
            logits = model(tokens[index], states[index], total_steps=n, backward_cap=cap)
            value = torch.nn.functional.cross_entropy(logits.flatten(0,1), targets[index].flatten(), reduction='none')
            (value.mul(masks[index]).sum()*(factor/count)*8192).backward()
    for parameters in zip(expected.parameters(), *[m.parameters() for m in replicas], strict=True):
        reference, *workers = parameters
        combined = sum(p.grad for p in workers)/2
        assert torch.allclose(reference.grad/8192, combined/8192, rtol=3e-5, atol=1e-6)


def test_unscale_precedes_clipping_and_missing_grad_prevents_update():
    c = ParcaeCandidateConfig(dim=32, heads=2, ffn_dim=64, vocab_size=128)
    model = ParcaeCandidate(c)
    optimizer = ParcaeOptimizerCandidate(model, gate_policy='release_muon')
    for p in model.parameters():
        p.grad = torch.ones_like(p)*8192
    true_norm = sum(p.numel() for p in model.parameters())**.5
    assert abs(finish_gradients(model, scale=8192)-true_norm) < 1e-4
    assert abs(sum(float(p.grad.square().sum()) for p in model.parameters())**.5-1) < 1e-5
    configure_optimizer(optimizer, 40)
    assert all(g['lr'] == 1e-4 for g in optimizer.param_groups)
    next(model.parameters()).grad = None
    with pytest.raises(ValueError):
        finish_gradients(model, scale=1)


def test_resume_rejects_changed_geometry_or_inconsistent_cursor():
    signature = {'world':32, 'accumulation':16, 'precision':'fp32'}
    state = {'qualification_only':True,'signature':signature,'update':2,'cursor':1024,
             'real_targets':1_000_000,'model_matmuls':100,'optimizer_matmuls':10}
    validate_resume(state, signature)
    with pytest.raises(ValueError, match='signature'):
        validate_resume(state, {**signature,'accumulation':8})
    with pytest.raises(ValueError, match='cursor'):
        validate_resume({**state,'cursor':768}, signature)


def test_shared_validated_mapping_preserves_masks_counts_and_windows(tmp_path):
    documents = [{'id':str(i),'split':'train','domain':domain,'tokens':list(range(4,19+i))}
                 for i,domain in enumerate(['web','math','code'])]
    original = PackedCorpus(documents,8,23)
    path = tmp_path/'packed'
    original.save_mapped(path,'source-fixture')
    verified = PackedCorpus.from_mapped(path,8,23,'source-fixture')
    candidate = map_after_shared_validation(path,file_sha256(path/'manifest.json'))
    assert candidate.counts(0,candidate.windows) == verified.counts(0,verified.windows)
    left,_ = candidate.batch(list(range(candidate.windows)))
    right,_ = verified.batch(list(range(verified.windows)))
    assert torch.equal(left.inputs,right.inputs)
    assert torch.equal(left.targets,right.targets)
    assert torch.equal(left.target_mask,right.target_mask)
    with pytest.raises(ValueError, match='manifest'):
        map_after_shared_validation(path,'changed')
