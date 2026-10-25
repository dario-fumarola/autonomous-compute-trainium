import copy

import pytest
import torch

from compute_trainium.low_depth_recovery import gradient_identity, rebind_checkpoint
from compute_trainium.owned_native_phase import create_worker
from compute_trainium.parcae_scientific_state import signature


def test_all_declared_gradient_stages_pass_actual_worker_identity_validation(tmp_path):
    names = []
    for arm in ['ordinary', 'fixed1', 'poisson2']:
        for label in ['stable1B', 'matched', 'equal']:
            identity = gradient_identity(arm, label); names.append(identity)
            # A valid identity reaches the missing setup file, without Docker.
            with pytest.raises(FileNotFoundError):
                create_worker(tmp_path, identity+'-'+arm, 0)
    assert len(set(names)) == 9
    with pytest.raises(ValueError, match='safe phase identity'):
        create_worker(tmp_path, 'low-depth-main-gradient-v1-fixed1-stable1B-fixed1', 0)


def test_adoption_changes_only_plan_signature_and_preserves_tensor_aliases():
    case = {'arm':'fixed1','seed':1231,'state_seed':3240,'initial_state':'family',
            'depth_table_sha256':'depth','peak_lr':.016,'total_updates':6872}
    plan = {'purpose':'natural_low_depth_scientific_pilot','model':{'dim':32},
            'precision':'fp16','accumulation':16,'routing_seed':271828,
            'data':{'packing_fingerprint':'packed','corpus_sha256':'source'},
            'initial_states':{'family':{'sha256':'init'}},'cases':{'case':case}}
    tensor = torch.arange(8, dtype=torch.float32)
    saved = {'case_id':'case','signature':signature('old',plan,case),
             'model':{'weight':tensor},'optimizer':{'moment':tensor.clone()},
             'cpu_rng_by_rank':[torch.arange(5,dtype=torch.uint8)],'cursor':1920*512}
    new_plan = copy.deepcopy(plan)
    result = rebind_checkpoint(saved,'old',plan,'new',new_plan,'case')
    assert saved['signature']['plan_sha256']=='old'
    assert result['signature']=={**saved['signature'],'plan_sha256':'new'}
    for key in saved.keys()-{'signature'}:
        assert result[key] is saved[key]
    new_plan['cases']['case']['peak_lr']=.008
    with pytest.raises(ValueError,match='scientific configuration'):
        rebind_checkpoint(saved,'old',plan,'new',new_plan,'case')
