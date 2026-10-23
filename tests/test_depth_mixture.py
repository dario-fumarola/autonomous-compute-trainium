"""Routing, resource and global-gradient contracts for the new depth mixture."""
import copy
from dataclasses import replace

import pytest
import torch

from compute_trainium.depth_mixture import rank_depths, routed_indices, window_permutation, mixture_step_flops
from compute_trainium.preln_recurrence import PreNormRecurrenceConfig


def test_depth_mixture_has_exact_mean_two_and_unchanged_unique_parameters():
    for world in [3, 4, 8, 32, 64]:
        depths=rank_depths(world,'mixture')
        assert len(depths)==world and sum(depths)==2*world
        assert depths.count(1)==depths.count(3)>0
    config=PreNormRecurrenceConfig()
    assert len({replace(config,repetitions=r).parameter_count() for r in [1,2,3]})==1
    for world in [32,64]:
        actual=mixture_step_flops(config,world,4,2,1024)
        expected=config.flops(1024,1024,world*4*2)['training_matmul_flops']
        assert actual==expected


def test_routing_covers_exact_global_source_set_and_is_resume_deterministic():
    world,batch,accumulation=32,4,2
    schedules=[]
    for cursor in [0,256,512]:
        indices=[i for micro in range(accumulation) for rank in range(world)
                 for i in routed_indices(cursor,micro,rank,world,batch,accumulation)]
        assert sorted(indices)==list(range(cursor,cursor+256))
        assert len(set(indices))==256
        schedules.append(indices)
    window_permutation.cache_clear()
    resumed=[i for micro in range(accumulation) for rank in range(world)
             for i in routed_indices(512,micro,rank,world,batch,accumulation)]
    assert resumed==schedules[-1]
    assert [i-256 for i in schedules[1]]!=schedules[0]


def test_mixed_depth_ddp_average_equals_global_real_target_gradient():
    # Numerical software fixture: no generated research examples.
    torch.manual_seed(104)
    core=torch.nn.Linear(3,3,bias=False).double()
    x=torch.randn(12,3,dtype=torch.double);targets=torch.randn_like(x)
    masks=torch.tensor([1,0,1,1,1,0,1,1,1,0,1,1],dtype=torch.double)
    depths=rank_depths(3,'mixture');total=masks.sum()
    worker_gradients=[]
    for rank,depth in enumerate(depths):
        model=copy.deepcopy(core);loss=torch.zeros((),dtype=torch.double)
        for micro in range(2):
            ids=routed_indices(0,micro,rank,3,2,2)
            state=torch.zeros_like(x[ids])
            for _ in range(depth):state=torch.tanh(model(state+x[ids]))
            loss=loss+((state-targets[ids]).square().sum(-1)*masks[ids]).sum()
        (loss*3/total).backward();worker_gradients.append(model.weight.grad)
    reference=copy.deepcopy(core);loss=torch.zeros((),dtype=torch.double)
    for micro in range(2):
        for rank,depth in enumerate(depths):
            for i in routed_indices(0,micro,rank,3,2,2):
                state=torch.zeros_like(x[i])
                for _ in range(depth):state=torch.tanh(reference(state+x[i]))
                loss=loss+(state-targets[i]).square().sum()*masks[i]/total
    loss.backward()
    torch.testing.assert_close(torch.stack(worker_gradients).mean(0),reference.weight.grad,rtol=1e-12,atol=1e-12)
    assert reference.weight.grad.norm()>0


def test_invalid_depth_and_routing_inputs_fail_closed():
    with pytest.raises(ValueError):rank_depths(2,'mixture')
    with pytest.raises(ValueError):rank_depths(32,'unknown')
    with pytest.raises(ValueError):routed_indices(0,2,0,32,4,2)
    with pytest.raises(ValueError):window_permutation(-1,256,23)


def test_new_test_gate_rejects_unknown_checkpoint_depth_or_evaluator():
    from compute_trainium.depth_evaluation_gate import validate_depth_test_release
    release={'status':'FROZEN_VARIABLE_DEPTH_TEST_RELEASE','corpus_sha256':'fresh',
             'evaluator_sha256':'evaluator','models':[{'checkpoint_sha256':'model','allowed_depths':[1,2,3,4]}]}
    args=dict(corpus_sha='fresh',checkpoint_sha='model',repetitions=3,evaluator_sha='evaluator')
    validate_depth_test_release(release,**args)
    for key,bad in [('corpus_sha','consumed'),('checkpoint_sha','other'),('repetitions',5),('evaluator_sha','changed')]:
        with pytest.raises(ValueError):validate_depth_test_release(release,**dict(args,**{key:bad}))


def test_three_contrast_family_interval_is_wider_and_recovers_normal_interval():
    from compute_trainium.depth_analysis import family_seed_interval
    gains={'103':.02,'211':.03,'307':.04}
    single=family_seed_interval(gains,comparisons=1)
    family=family_seed_interval(gains,comparisons=3)
    assert single['bonferroni_family_interval_95']==pytest.approx(single['interval_95'])
    assert family['bonferroni_family_interval_95'][0]<single['interval_95'][0]
    assert family['bonferroni_family_interval_95'][1]>single['interval_95'][1]


def test_explicit_optimizer_groups_cover_embedding_norm_and_matrix_once():
    from compute_trainium.pretraining_optimizer import parameter_groups
    model=torch.nn.ModuleDict({'embedding':torch.nn.Embedding(16,4),
           'norm':torch.nn.LayerNorm(4),'linear':torch.nn.Linear(4,4,bias=False)})
    groups=parameter_groups(model,model['embedding'].weight)
    decayed={id(p) for p in groups[0]['params']};undecayed={id(p) for p in groups[1]['params']}
    assert decayed=={id(model['linear'].weight)}
    assert undecayed=={id(model['embedding'].weight),id(model['norm'].weight),id(model['norm'].bias)}
    assert not decayed&undecayed
    assert [g['weight_decay'] for g in groups]==[.1,0.]
