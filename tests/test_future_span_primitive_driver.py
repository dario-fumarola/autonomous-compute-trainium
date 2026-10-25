import pytest
from compute_trainium.future_span_primitive_driver import worker_sources
from compute_trainium.native_snapshot_manifest import split_snapshot_sources


def test_primitive_driver_freezes_host_reference_without_requiring_uncopied_snapshot_file():
    sources={'scripts/train_future_span_scientific.py':'trainer','compute_trainium/future_span.py':'model',
             'third_party/parcae_reference/muon_adamw.py':'host-reference'}
    worker,_=split_snapshot_sources(sources)
    plan={'source_sha256':sources,'worker_source_sha256':worker,
          'execution_variant':'fp16_body_fp32_attention_primitives'}
    assert worker_sources(plan)==worker
    with pytest.raises(ValueError,match='snapshot'):
        worker_sources({**plan,'worker_source_sha256':sources})
    with pytest.raises(ValueError,match='Explicit'):
        worker_sources({**plan,'execution_variant':'installed_sdpa'})
