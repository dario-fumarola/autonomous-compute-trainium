import pytest

from compute_trainium.scale_pilot_size import profiled_inventory


def test_joint_scale_profile_has_strong_parameter_ceiling_control_and_fixed_operator_shapes():
    recurrent,ordinary=profiled_inventory('medium')
    assert recurrent['parameters']==375940224
    assert ordinary['parameters']==374888960<recurrent['parameters']
    for row in [recurrent,ordinary]:
        config=row['config']
        assert config['dim']//config['heads']==128
        assert config['dim']%128==0 and config['ffn_dim']%128==0
        assert row['adamw_parameters']+row['muon_parameters']==row['parameters']
    with pytest.raises(ValueError):profiled_inventory('another-search')
