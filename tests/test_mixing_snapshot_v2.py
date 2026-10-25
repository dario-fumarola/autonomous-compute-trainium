from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from scripts.train_mixing_control_scientific_v2 import worker_sources


def test_control_worker_freeze_retains_cpu_reference_without_expecting_it_in_snapshot():
    sources={'scripts/train_future_span_scientific.py':'engine',
        'scripts/train_mixing_control_scientific_v2.py':'wrapper',
        'compute_trainium/parcae_mixing_control.py':'model',
        'third_party/parcae_reference/muon_adamw.py':'independent-cpu-reference',
        'tests/test_mixing_snapshot_v2.py':'test'}
    worker,host=split_snapshot_sources(sources)
    plan={'source_sha256':sources,'worker_source_sha256':worker,
          'execution_variant':'fp16_body_fp32_attention_primitives'}
    assert worker_sources(plan)==worker
    assert 'third_party/parcae_reference/muon_adamw.py' in host
    assert 'third_party/parcae_reference/muon_adamw.py' not in worker
