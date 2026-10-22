"""Small-vocabulary models from the pinned TorchTitan Neuron fork."""

import torch
import torch.nn.functional as F
from torch import nn


def build_titan(c):
    # These factories are private upstream APIs; pin and record the fork commit.
    from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm
    from torchtitan.models.llama3 import (
        _build_llama3_layers, _EMBEDDING_SKIP_INIT, _NORM_INIT, _output_linear_init,
    )
    from torchtitan.models.llama3.model import Llama3Model

    config = Llama3Model.Config(
        dim=c.dim, vocab_size=c.vocab_size, enable_weight_tying=True,
        tok_embeddings=Embedding.Config(num_embeddings=c.vocab_size, embedding_dim=c.dim,
                                        param_init=_EMBEDDING_SKIP_INIT),
        lm_head=Linear.Config(in_features=c.dim, out_features=c.vocab_size,
                              param_init=_output_linear_init(c.dim)),
        norm=RMSNorm.Config(normalized_shape=c.dim, param_init=_NORM_INIT),
        layers=_build_llama3_layers(
            n_layers=c.layers, dim=c.dim, n_heads=c.heads, n_kv_heads=c.kv_heads,
            hidden_dim=c.ffn_dim, attn_backend="sdpa", fuse_qkv=True,
            rope=ComplexRoPE.Config(dim=c.dim // c.heads, max_seq_len=c.max_seq,
                                   theta=10000, use_prefix_cache_for_positions=True),
        ),
    )
    model = config.build()
    model.init_states()
    model._skip_lm_head = True

    class RealTargetModel(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
            self.config = c

        def forward(self, tokens, prediction_positions=None, attention_mask=None, prediction_spans=None):
            h = self.decoder(tokens, attention_masks=attention_mask)
            if prediction_spans is not None:
                h = torch.cat([h[:, a:b] for a, b in prediction_spans], dim=1)
            elif prediction_positions is not None:
                h = h.index_select(1, prediction_positions)
            return F.linear(h, self.decoder.lm_head.weight)

    return RealTargetModel(model)
