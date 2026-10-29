"""Explicit graph choice for prospective registered interleaved learning runs."""
from .model import ModelConfig
from .interleaved_workspace import InterleavedWorkspace
from .interleaved_static_positions import StaticPositionInterleavedWorkspace


def make_model(plan, *, explicit_attention=False):
    implementation=plan['interleaved_implementation']
    if implementation not in ('original','static_positions'):
        raise ValueError('Explicit qualified execution implementation required')
    cls=(StaticPositionInterleavedWorkspace
         if explicit_attention and implementation=='static_positions'
         else InterleavedWorkspace)
    return cls(ModelConfig(**plan['model_config']),explicit_attention=explicit_attention)
