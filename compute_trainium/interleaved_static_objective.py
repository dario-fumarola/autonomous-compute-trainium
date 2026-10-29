"""Fixed-position Native layout with the original independent CPU reference."""
from .model import ModelConfig
from .interleaved_workspace import InterleavedWorkspace
from .interleaved_static_positions import StaticPositionInterleavedWorkspace


def make_model(plan, *, explicit_attention=False):
    cls=StaticPositionInterleavedWorkspace if explicit_attention else InterleavedWorkspace
    return cls(ModelConfig(**plan['model_config']),explicit_attention=explicit_attention)
