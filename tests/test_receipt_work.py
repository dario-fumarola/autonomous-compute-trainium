import pytest
from compute_trainium.receipt_work import estimate_matmuls


def test_training_probe_is_a_microbatch_not_an_extra_optimizer_update():
    work=estimate_matmuls({'updates':10,'accumulation':2,'cpu_probe':{'passed':True},
        'flops_per_global_step':{'training_matmul_flops':800}})
    assert work['optimizer_update_matmul_flops']==8000
    assert work['known_training_probe_matmul_flops']==400


def test_sharded_readout_adds_probes_but_scalar_reuses_first_scored_window():
    # Numerical metadata fixtures only, never training/evaluation examples.
    docs=[{'id':'a','split':'dev','domain':'web','cohort':'curve','tokens':[0]*6},
        {'id':'b','split':'dev','domain':'math','cohort':'curve','tokens':[0]*20}]
    training={'signature':{'context':4,'world':2,'batch':2,'accumulation':2},
        'flops_per_global_step':{'forward_matmul_flops':400}}
    scalar={'status':'PASS_CHECKPOINT_DEV_EVALUATION','split':'dev','cohort':'curve',
        'max_documents_per_domain':0,'documents':2,'targets':26}
    a=estimate_matmuls(scalar,scalar_training_report=training,dev_documents=docs)
    assert a['readout_matmul_flops']==350  # 2+5 windows, 50 FLOPs each.
    b=estimate_matmuls({'targets':26,'scored_windows':7,'native_probe_windows':2,'forward_flops_per_window':50})
    assert b['readout_matmul_flops']==450
    with pytest.raises(ValueError):
        estimate_matmuls(dict(scalar,targets=27),scalar_training_report=training,dev_documents=docs)
    assert estimate_matmuls(scalar)['readout_matmul_flops'] is None
