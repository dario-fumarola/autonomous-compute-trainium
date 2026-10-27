"""Canonical flat prediction-state container; tensor contents stay exact."""
from collections.abc import Mapping
from scripts.qualify_native import digest


def canonical_prediction_checkpoint(value):
    if value.get('purpose')!='scale_retained_prediction_weights' or not isinstance(value.get('model'),Mapping):
        raise ValueError('Exact retained-prediction schema required')
    result={**value,'model':dict(value['model'])}
    if digest(result['model'])!=value['model_digest']:
        raise AssertionError('Every prediction tensor/name/dtype/shape must match the pre-save digest')
    return result
