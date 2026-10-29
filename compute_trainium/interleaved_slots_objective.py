"""Native-only layout alternative; independent CPU references keep old layout."""
from .model import ModelConfig
from .interleaved_workspace import InterleavedWorkspace
from .interleaved_slots import SlotInterleavedWorkspace


def make_model(plan, *, explicit_attention=False):
    cls=SlotInterleavedWorkspace if explicit_attention else InterleavedWorkspace
    return cls(ModelConfig(**plan['model_config']),explicit_attention=explicit_attention)
