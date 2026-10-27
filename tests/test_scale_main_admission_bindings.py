import pytest
from scripts.run_scale_main import validate_storage_binding,validate_body_scope_inputs


def test_main_rejects_stale_storage_proof_and_relaxed_reserves():
    proof=dict(tuning_report_sha256='selected',scope_freeze_sha256='8B',boundary_numerical_proof_sha256='qualified',
               protected_host_bytes=512*1024**3,protected_filesystem_bytes=35*1024**3,
               native_worker_allowance_bytes=192*1024**3,new_ram_allowance_bytes=0)
    expected=dict(tuning_sha256='selected',scope_sha256='8B',boundary_proof_sha256='qualified')
    validate_storage_binding(proof,**expected)
    for key,value in [('tuning_report_sha256','older-grid'),('scope_freeze_sha256','old-3.75B'),
                       ('boundary_numerical_proof_sha256','old-precision'),
                       ('protected_host_bytes',511*1024**3),('protected_filesystem_bytes',34*1024**3),
                       ('native_worker_allowance_bytes',128*1024**3),('new_ram_allowance_bytes',1024**3)]:
        with pytest.raises(AssertionError):validate_storage_binding({**proof,key:value},**expected)


def test_body_scope_rejects_repacked_data_and_altered_pristine_initializer_receipt():
    frozen=dict(accepted_packing_manifest_sha256='accepted',packing_fingerprint='source-order',
                original_unused_initializer_report_sha256='zero-updates',user_clock_amendment_sha256='owner')
    expected=dict(manifest_sha256='accepted',packing_fingerprint='source-order',initializer_report_sha256='zero-updates',clock_sha256='owner')
    validate_body_scope_inputs(frozen,**expected)
    for key in expected:
        with pytest.raises(AssertionError):validate_body_scope_inputs(frozen,**{**expected,key:'changed'})
