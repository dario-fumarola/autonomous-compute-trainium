"""Snapshot policy for the explicitly registered primitive-attention driver."""
from .native_snapshot_manifest import split_snapshot_sources


def worker_sources(plan):
    if plan.get('execution_variant')!='fp16_body_fp32_attention_primitives':
        raise ValueError('Explicit primitive-attention execution plan required')
    expected,_=split_snapshot_sources(plan['source_sha256'])
    if ('scripts/train_future_span_scientific.py' not in expected
            or plan['worker_source_sha256']!=expected):
        raise ValueError('Exact frozen executable snapshot required')
    return expected
