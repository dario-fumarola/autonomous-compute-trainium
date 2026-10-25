import copy

import pytest
import torch

from compute_trainium.future_span_scientific_state import SCIENCE, signature
from compute_trainium.future_span_transport import transport_stable
from compute_trainium.parcae_scientific_state import schedule_identity
from scripts.qualify_native import digest


def fixture():
    # Serialized-state software fixture only, not a research model or example.
    plan = {'purpose': SCIENCE, 'model': {'dim': 768},
        'future_config': {'context': 1024, 'horizon': 32, 'stride': 16, 'weight': .1},
        'gradient_rule': 'all_sampled_steps', 'initial_states': {'fresh': {'sha256': 'initial'}},
        'data': {'packing_fingerprint': 'pack', 'corpus_sha256': 'corpus'},
        'auxiliary_data': {'prefix_sha256': 'anchors'}, 'precision': 'fp16_body_fp32_heads_masters',
        'accumulation': 16, 'routing_seed': 271828}
    case = {'cell': 'one-ntp', 'arm': 'one', 'objective_weight': 0., 'seed': 5231,
        'state_seed': 7240, 'initial_state': 'fresh', 'depth_table_sha256': 'depth',
        'work_table_sha256': 'work', 'total_updates': 40, 'peak_lr': .032,
        'warmup_targets': 100, 'cooldown_start_update': 20,
        'cooldown_start_targets': 1000, 'end_targets': 2000}
    plan['cases'] = {'parent': case}
    destination = copy.deepcopy(plan)
    destination['cases'] = {'child': {**case, 'total_updates': 60, 'end_targets': 3000,
        'cooldown_start_update': 30, 'cooldown_start_targets': 1500,
        'parent': {'case_id': 'parent', 'update': 20}}}
    model = {'weight': torch.tensor([1., 2.])}
    optimizer = {'step': 20, 'moment': torch.tensor([.1, .2])}
    saved = {'purpose': SCIENCE, 'qualification_only': False,
        'signature': signature('a'*64, plan, case), 'case_id': 'parent',
        'schedule': schedule_identity(case), 'stable_trunk': True,
        'update': 20, 'cursor': 20*512, 'real_targets': 1000,
        'domain_targets': [600, 200, 200], 'estimated_dense_matrix_ops': 123456,
        'valid_anchors': 111, 'future_source_position_presentations': 111*31,
        'executed_head_slots': 20*512*63, 'model': model, 'optimizer': optimizer,
        'cpu_rng_by_rank': [torch.tensor([i], dtype=torch.uint8) for i in range(32)],
        'model_optimizer_digest': digest({'model': model, 'optimizer': optimizer})}
    counters = dict(expected_targets=1000, expected_domains=[600, 200, 200],
                    expected_matrix_ops=123456, expected_anchors=111)
    options = dict(source_plan=plan, source_plan_sha='a'*64, source_case_id='parent',
        source_checkpoint_sha='c'*64, destination_plan=destination,
        destination_plan_sha='b'*64, destination_case_id='child', expected_counters=counters)
    return saved, options


def test_transport_preserves_every_tensor_history_and_original_plan_binding():
    saved, options = fixture()
    before = digest(saved)
    adopted, proof = transport_stable(saved, **options)
    assert digest(saved) == before
    assert adopted['signature']['plan_sha256'] == 'b'*64
    assert saved['signature']['plan_sha256'] == 'a'*64
    assert adopted['case_id'] == 'parent' and adopted['schedule'] == saved['schedule']
    assert adopted['model'] is saved['model'] and adopted['optimizer'] is saved['optimizer']
    assert adopted['cpu_rng_by_rank'] is saved['cpu_rng_by_rank']
    assert proof['inherited_matrix_ops'] == 123456
    assert adopted['continuation_provenance'] == [proof]


@pytest.mark.parametrize('field,value', [('seed', 7), ('objective_weight', .1),
    ('state_seed', 8), ('depth_table_sha256', 'changed'), ('work_table_sha256', 'changed')])
def test_transport_cannot_change_the_scientific_computation(field, value):
    saved, options = fixture()
    options['destination_plan']['cases']['child'][field] = value
    with pytest.raises(ValueError):
        transport_stable(saved, **options)


def test_transport_rejects_reset_history_annealing_and_unregistered_parent():
    saved, options = fixture()
    saved['optimizer']['moment'].zero_()
    with pytest.raises(ValueError, match='optimizer'):
        transport_stable(saved, **options)
    saved, options = fixture()
    saved['stable_trunk'] = False
    with pytest.raises(ValueError):
        transport_stable(saved, **options)
    saved, options = fixture()
    saved['qualification_only'] = True
    with pytest.raises(ValueError):
        transport_stable(saved, **options)
    saved, options = fixture()
    options['destination_plan']['cases']['child']['parent']['update'] = 19
    with pytest.raises(ValueError, match='parent'):
        transport_stable(saved, **options)
    saved, options = fixture()
    options['destination_plan']['cases']['child']['peak_lr'] = .016
    with pytest.raises(ValueError, match='learning recipe'):
        transport_stable(saved, **options)
