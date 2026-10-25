import json
import pytest

from scripts.evaluate_natural_code_context_recovered import require_endpoint_gradient


def test_explicit_recovery_receipt_path_checks_the_same_exact_checkpoint_gate(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    file=tmp_path/'explicit-recovery.json'
    row={'arm':'poisson2','readout':'matched','independent_cpu':{
        'status':'PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT','checkpoint_sha256':'qualified-checkpoint'}}
    file.write_text(json.dumps({'records':[row]}))
    # No legacy results/main directory exists. Only the explicit recovered
    # receipt is eligible;the numerical/identity requirement is unchanged.
    assert require_endpoint_gradient(file,'matched','qualified-checkpoint')==row
    with pytest.raises(ValueError):require_endpoint_gradient(file,'equal','qualified-checkpoint')
    with pytest.raises(ValueError):require_endpoint_gradient(file,'matched','different-checkpoint')
    file.write_text(json.dumps({'records':[row,row]}))
    with pytest.raises(ValueError):require_endpoint_gradient(file,'matched','qualified-checkpoint')
    row['independent_cpu']['status']='FAIL_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT'
    file.write_text(json.dumps({'records':[row]}))
    with pytest.raises(ValueError):require_endpoint_gradient(file,'matched','qualified-checkpoint')
