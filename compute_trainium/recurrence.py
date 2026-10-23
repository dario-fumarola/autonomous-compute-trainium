"""Fixed depth recurrence with input injection and sandwich normalization.

The topology follows the additive-injection/zero-state options in
seal-rg/recurrent-pretraining (1ea7220e), not the full Huginn training recipe.
There is no sampling, truncated gradient, learned exit, or auxiliary objective.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import ModelConfig


@dataclass(frozen=True)
class RecurrenceConfig(ModelConfig):
    prelude: int = 2
    core: int = 16
    coda: int = 2
    repetitions: int = 2
    untied: bool = False

    def __post_init__(self):
        super().__post_init__()
        if (min(self.prelude, self.core, self.coda, self.repetitions) < 1
                or self.prelude + self.core + self.coda != self.layers):
            raise ValueError("Positive prelude/core/coda must partition physical layers")

    @property
    def effective_layers(self):
        return self.prelude + self.repetitions*self.core + self.coda

    def parameter_count(self):
        physical = self.effective_layers if self.untied else self.layers
        base = {k: getattr(self, k) for k in ModelConfig.__dataclass_fields__}
        base["layers"] = physical
        return ModelConfig(**base).parameter_count() + 2*self.dim*physical

    def flops(self, real_targets, expanded_positions, batch=1):
        base = {k: getattr(self, k) for k in ModelConfig.__dataclass_fields__}
        base["layers"] = self.effective_layers
        result = ModelConfig(**base).flops(real_targets, expanded_positions, batch)
        result["effective_layers"] = self.effective_layers
        result["note"] = "All recurrent applications counted; full BPTT, no activation recomputation. Injection and RMSNorm are elementwise, excluded as in baseline."
        return result


def fixed_unroll(embedded, layers, config, apply_layer, extra_layers=()):
    """Shared by the native wrapper and independent CPU block implementation."""
    e = embedded
    for i in range(config.prelude):
        e = apply_layer(layers[i], e, i)
    state = torch.zeros_like(e)
    for iteration in range(config.repetitions):
        state = state + e
        for j in range(config.core):
            i = config.prelude + j
            if config.untied and iteration:
                extra = (iteration-1)*config.core+j
                state = apply_layer(extra_layers[extra], state, config.layers+extra)
            else:
                state = apply_layer(layers[i], state, i)
    for i in range(config.prelude+config.core, config.layers):
        state = apply_layer(layers[i], state, i)
    return state


def build_recurrent(config):
    from .titan import build_titan
    base = {k: getattr(config, k) for k in ModelConfig.__dataclass_fields__}
    decoder = build_titan(ModelConfig(**base)).decoder

    class FixedRecurrentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.decoder = decoder
            # Parameter sharing is by repeated calls, never copied parameters.
            self.extra_layers = nn.ModuleList([
                copy.deepcopy(decoder.layers[str(config.prelude+j)])
                for _ in range(config.repetitions-1 if config.untied else 0)
                for j in range(config.core)])
            physical = config.effective_layers if config.untied else config.layers
            self.post_attention = nn.ModuleList([nn.RMSNorm(config.dim, eps=1e-5) for _ in range(physical)])
            self.post_ffn = nn.ModuleList([nn.RMSNorm(config.dim, eps=1e-5) for _ in range(physical)])
            if sum(p.numel() for p in self.parameters()) != config.parameter_count():
                raise AssertionError("Unique parameter count differs from resource model")

        def hidden(self, tokens, attention_mask=None):
            def apply(block, h, index):
                h = self.post_attention[index](h + block.attention(block.attention_norm(h), attention_mask, None))
                return self.post_ffn[index](h + block.feed_forward(block.ffn_norm(h)))
            h = fixed_unroll(self.decoder.tok_embeddings(tokens), list(self.decoder.layers.values()),
                             config, apply, self.extra_layers)
            return self.decoder.norm(h)

        def forward(self, tokens, prediction_positions=None, attention_mask=None, prediction_spans=None):
            h = self.hidden(tokens, attention_mask)
            if prediction_spans is not None:
                h = torch.cat([h[:, a:b] for a, b in prediction_spans], dim=1)
            elif prediction_positions is not None:
                h = h.index_select(1, prediction_positions)
            return F.linear(h, self.decoder.lm_head.weight)

    return FixedRecurrentModel()


def config_from_signature(signature):
    cls = RecurrenceConfig if "repetitions" in signature["model"] else ModelConfig
    return cls(**signature["model"])


def build_from_config(config):
    if isinstance(config, RecurrenceConfig):
        return build_recurrent(config)
    from .titan import build_titan
    return build_titan(config)
