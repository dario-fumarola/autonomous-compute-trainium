"""Minimal recurrence around the already healthy pre-norm ordinary model.

One application is exactly the preserved ordinary forward function, with the
same weights and initialization. Input injection remains on every core pass.
"""
import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import ModelConfig
from .recurrence import RecurrenceConfig, fixed_unroll


@dataclass(frozen=True)
class PreNormRecurrenceConfig(RecurrenceConfig):
    normalization: str = 'pre'

    def __post_init__(self):
        super().__post_init__()
        if self.normalization!='pre':raise ValueError('This model preserves the qualified pre-norm blocks')

    def parameter_count(self):
        base={k:getattr(self,k) for k in ModelConfig.__dataclass_fields__}
        if self.untied:base['layers']=self.effective_layers
        return ModelConfig(**base).parameter_count()


def build_preln_recurrent(config):
    from .titan import build_titan
    base={k:getattr(config,k) for k in ModelConfig.__dataclass_fields__}
    decoder=build_titan(ModelConfig(**base)).decoder

    class PreNormRecurrentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config=config
            self.decoder=decoder
            self.extra_layers=nn.ModuleList([
                copy.deepcopy(decoder.layers[str(config.prelude+j)])
                for _ in range(config.repetitions-1 if config.untied else 0)
                for j in range(config.core)])
            if sum(p.numel() for p in self.parameters())!=config.parameter_count():
                raise AssertionError('Unique parameter count differs')

        def hidden(self,tokens,attention_mask=None):
            def apply(block,h,index):return block(h,attention_mask,None)
            h=fixed_unroll(self.decoder.tok_embeddings(tokens),list(self.decoder.layers.values()),
                           config,apply,self.extra_layers)
            return self.decoder.norm(h)

        def forward(self,tokens,prediction_positions=None,attention_mask=None,prediction_spans=None):
            h=self.hidden(tokens,attention_mask)
            if prediction_spans is not None:
                h=torch.cat([h[:,a:b] for a,b in prediction_spans],dim=1)
            elif prediction_positions is not None:h=h.index_select(1,prediction_positions)
            return F.linear(h,self.decoder.lm_head.weight)

    return PreNormRecurrentModel()
