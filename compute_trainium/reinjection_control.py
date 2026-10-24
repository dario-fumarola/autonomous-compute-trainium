"""A trained inexpensive control for recurrence's boundary addition.

One core application, then the same prelude-state addition used before a
second recurrent application, then the coda. No extra matrices or targets.
This is a separate model identity; existing recurrence forwards are unchanged.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import ModelConfig
from .preln_recurrence import PreNormRecurrenceConfig
from .independent import IndependentModel


@dataclass(frozen=True)
class ReinjectionControlConfig(PreNormRecurrenceConfig):
    repetitions: int = 1
    reinjection_only: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.repetitions != 1 or self.untied or not self.reinjection_only:
            raise ValueError('The cheap control executes each physical block once')


def reinjection_unroll(embedded, layers, config, apply_layer):
    e = embedded
    for i in range(config.prelude):
        e = apply_layer(layers[i], e, i)
    h = e
    for i in range(config.prelude, config.prelude+config.core):
        h = apply_layer(layers[i], h, i)
    h = h+e
    for i in range(config.prelude+config.core, config.layers):
        h = apply_layer(layers[i], h, i)
    return h


def build_reinjection_control(config):
    from .titan import build_titan
    if not isinstance(config, ReinjectionControlConfig):
        raise ValueError('Explicit cheap-control configuration required')
    base = {k: getattr(config, k) for k in ModelConfig.__dataclass_fields__}
    decoder = build_titan(ModelConfig(**base)).decoder

    class ReinjectionControl(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.decoder = decoder
            if sum(p.numel() for p in self.parameters()) != config.parameter_count():
                raise AssertionError('Parameter count differs from ordinary')

        def hidden(self, tokens, attention_mask=None):
            h = reinjection_unroll(self.decoder.tok_embeddings(tokens),
                list(self.decoder.layers.values()), config,
                lambda block, state, _: block(state, attention_mask, None))
            return self.decoder.norm(h)

        def forward(self, tokens, prediction_positions=None,
                    attention_mask=None, prediction_spans=None):
            h = self.hidden(tokens, attention_mask)
            if prediction_spans is not None:
                h = torch.cat([h[:, a:b] for a, b in prediction_spans], dim=1)
            elif prediction_positions is not None:
                h = h.index_select(1, prediction_positions)
            return F.linear(h, self.decoder.lm_head.weight)

    return ReinjectionControl()


class IndependentReinjectionControl(IndependentModel):
    """Separate CPU blocks/projections, with the control's explicit boundary."""
    def __init__(self, config, repetitions=1, prelude=2, core=16, coda=2):
        if repetitions != 1:
            raise ValueError('This control has one physical core application')
        super().__init__(config, repetitions, prelude, core, coda)

    def hidden(self, tokens, attention_mask=None):
        n = tokens.shape[1]
        if n > self.config.max_seq:
            raise ValueError('Rotary cache exceeded')
        h = self.embedding(tokens)
        cos, sin = self.cos[:, :, :n].to(h.dtype), self.sin[:, :, :n].to(h.dtype)
        for i in range(self.prelude):
            h = self.blocks[i](h, cos, sin, attention_mask)
        source = h
        for i in range(self.prelude, self.prelude+self.core):
            h = self.blocks[i](h, cos, sin, attention_mask)
        h = h+source
        for i in range(self.prelude+self.core, self.config.layers):
            h = self.blocks[i](h, cos, sin, attention_mask)
        return self.norm(h)
