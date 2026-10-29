"""CPU-only streaming reference for the trained sparse execution graph.

One fixed slot per boundary, no router and no new training recipe. Appending
source tokens retains exact layer K/V across calls. Hardware execution and
latency remain separately unqualified.
"""
import torch
from torch.nn import functional as F

from .depth_causal_workspace import DepthCausalWorkspace


class ChunkedStreamingReference:
    def __init__(self, model, family, *, stride=128):
        if family not in ('ordinary', 'generated', 'constant'):
            raise ValueError('An explicit trained family is required')
        if type(stride) is not int or stride < 1:
            raise ValueError('Positive source stride required')
        if model.training or any(p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('Evaluation-only CPU reference; hardware path is unqualified')
        self.model, self.family, self.stride = model, family, stride
        self.cache = [[] for _ in model.blocks]
        self.source_positions = 0
        self.batch = None
        self.failed = False
        self.model_signature = self._signature()

    def _signature(self):
        return tuple((name, id(value), value._version, value.dtype, value.device)
                     for name, value in [*self.model.named_parameters(), *self.model.named_buffers()])

    def _validate(self, tokens):
        if self.failed:
            raise RuntimeError('Failed append invalidates this cache; construct a new session')
        if self.model.training or self._signature() != self.model_signature:
            raise RuntimeError('Weights, buffers, dtype and evaluation mode must stay fixed')
        if tokens.device.type != 'cpu' or tokens.dtype != torch.long or tokens.ndim != 2 or min(tokens.shape) < 1:
            raise ValueError('Nonempty two-dimensional CPU int64 source tokens required')
        if self.batch is not None and tokens.shape[0] != self.batch:
            raise ValueError('Batch membership and ordering must remain fixed')
        if self.source_positions + tokens.shape[1] > self.model.config.max_seq:
            raise ValueError('No implicit sliding window or rotary extrapolation')
        if int(tokens.min()) < 0 or int(tokens.max()) >= self.model.config.vocab_size:
            raise ValueError('Source IDs outside the fixed vocabulary')

    @torch.inference_mode()
    def append(self, tokens):
        """Return each next-token distribution using only its available prefix.

        Input chunking is arbitrary; workspace boundaries depend on total source
        position, not on API calls. The caller supplies only observed/generated
        source tokens, never expected next-token labels for selection.
        """
        self._validate(tokens)
        start = self.source_positions
        outputs = []
        try:
            left = 0
            while left < tokens.shape[1]:
                position = start+left
                n = min(self.stride-position % self.stride, tokens.shape[1]-left)
                pos = torch.arange(position, position+n, device=tokens.device)
                source = self.model.embeddings(tokens[:, left:left+n])
                hidden = DepthCausalWorkspace._cached_stage(self.model, source, self.cache, pos)
                if (position+n) % self.stride == 0 and self.family != 'ordinary':
                    state = hidden[:, -1:]
                    proposal = (state if self.family == 'generated' else
                        self.model.source_scale*self.model.embedding.weight.mean(0).view(1,1,-1).expand_as(state))
                    state = DepthCausalWorkspace._cached_stage(self.model, proposal, self.cache, pos[-1:])
                    hidden = torch.cat((hidden[:, :-1], state), 1)
                outputs.append(F.linear(hidden, self.model.prediction_weight))
                left += n
            logits = torch.cat(outputs, 1)
        except BaseException:
            self.failed = True
            raise
        self.source_positions += tokens.shape[1]
        self.batch = tokens.shape[0]
        return logits

    def cache_inventory(self):
        if self.failed:
            raise RuntimeError('Incomplete cache has no valid inventory')
        lengths = [sum(k.shape[2] for k, v, positions in layer) for layer in self.cache]
        expected = self.source_positions + (self.source_positions//self.stride if self.family != 'ordinary' else 0)
        if any(n != expected for n in lengths):
            raise AssertionError('Chronological source/latent coverage mismatch')
        return dict(source_positions=self.source_positions,
                    latent_positions=expected-self.source_positions,
                    stored_positions_per_layer=expected,
                    kv_tensor_bytes=sum(t.numel()*t.element_size() for layer in self.cache
                                        for k,v,positions in layer for t in (k,v)),
                    scope='Actual K/V tensor payload only; excludes positions, allocator, temporary concatenations and model weights. CPU reference, not serving memory or latency.')
