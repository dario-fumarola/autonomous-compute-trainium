"""Explicit graph choice for prospective registered interleaved learning runs."""
from .model import ModelConfig
from .interleaved_workspace import InterleavedWorkspace
from .interleaved_static_positions import StaticPositionInterleavedWorkspace
from .grouped_interleaved_workspace import GroupedInterleavedWorkspace


def make_model(plan, *, explicit_attention=False):
    implementation=plan['interleaved_implementation']
    if implementation not in ('original','static_positions','grouped'):
        raise ValueError('Explicit qualified execution implementation required')
    classes={'original':InterleavedWorkspace,'static_positions':StaticPositionInterleavedWorkspace,'grouped':GroupedInterleavedWorkspace}
    cls=classes[implementation] if explicit_attention else InterleavedWorkspace
    return cls(ModelConfig(**plan['model_config']),explicit_attention=explicit_attention)
