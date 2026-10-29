"""Grouped Native implementation with chronological independent CPU reference."""
from .model import ModelConfig
from .interleaved_workspace import InterleavedWorkspace
from .grouped_interleaved_workspace import GroupedInterleavedWorkspace

def make_model(plan,*,explicit_attention=False):
    cls=GroupedInterleavedWorkspace if explicit_attention else InterleavedWorkspace
    return cls(ModelConfig(**plan['model_config']),explicit_attention=explicit_attention)
