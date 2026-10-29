"""CPU reference: sparse persistent latent slots with exact differentiable KV.

Every full source chunk may append generated states. Later real tokens attend
to those states. No Jacobi solver, vocabulary sampling, target weighting or
training admission. Tied tables match the strongest existing ordinary family.
"""
import torch
from torch.nn import functional as F

from .budgeted_interleaved_reference import check_budget
from .depth_causal_workspace import DepthCausalWorkspace
from .tied_interleaved_ordinary import TiedOrdinary


def validate(stride, steps, mode):
    check_budget(steps)
    if type(stride) is not int or stride < 1 or mode not in ('latent', 'pause'):
        raise ValueError('Positive source stride and an explicit workspace mode required')


class ChunkedPersistentWorkspace(TiedOrdinary):
    def forward(self, tokens, stride=64, steps=1, mode='latent'):
        validate(stride, steps, mode)
        if steps == 0:
            # Exact existing full-sequence ordinary path, including gradients.
            return super().forward(tokens)
        return self.cached(tokens, stride, steps, mode)

    def cached(self, tokens, stride=64, steps=1, mode='latent', *, detach_cache=False,
               return_states=False):
        validate(stride, steps, mode)
        source = self.embeddings(tokens)
        cache = [[] for _ in self.blocks]
        reads, thoughts = [], []
        length = source.shape[1]
        for left in range(0, length, stride):
            right = min(left+stride, length)
            positions = torch.arange(left, right, device=tokens.device)
            h = DepthCausalWorkspace._cached_stage(self, source[:, left:right], cache, positions,
                                                   detach_cache=detach_cache)
            if right % stride == 0 and steps:
                read = h[:, :-1]
                state = h[:, -1:]
                for _ in range(steps):
                    proposal = state if mode == 'latent' else self.source_scale*self.embedding.weight.mean(0).view(1,1,-1).expand_as(state)
                    thoughts.append(proposal)
                    state = DepthCausalWorkspace._cached_stage(self, proposal, cache, positions[-1:],
                                                               detach_cache=detach_cache)
                reads.append(torch.cat((read,state),1))
            else:
                reads.append(h)
        hidden = torch.cat(reads,1)
        logits = F.linear(hidden,self.prediction_weight)
        return (logits,tuple(thoughts)) if return_states else logits

    def full_reference(self, tokens, stride=64, steps=1, mode='latent', *, return_states=False):
        """Independent full-prefix recomputation in chronological token order."""
        validate(stride, steps, mode)
        source = self.embeddings(tokens)
        pieces, positions, read_indices, thoughts = [], [], [], []
        stored = 0
        for left in range(0,source.shape[1],stride):
            right = min(left+stride,source.shape[1])
            pieces.append(source[:,left:right])
            positions.extend(range(left,right))
            read_indices.extend(range(stored,stored+right-left))
            stored += right-left
            if right % stride == 0:
                for _ in range(steps):
                    if mode == 'latent':
                        proposal = self.body_at(torch.cat(pieces,1),torch.tensor(positions,device=tokens.device))[:,-1:]
                    else:
                        proposal = self.source_scale*self.embedding.weight.mean(0).view(1,1,-1).expand(source.shape[0],1,-1)
                    pieces.append(proposal)
                    thoughts.append(proposal)
                    positions.append(right-1)
                    read_indices[-1] = stored
                    stored += 1
        hidden = self.body_at(torch.cat(pieces,1),torch.tensor(positions,device=tokens.device))
        selected = hidden.index_select(1,torch.tensor(read_indices,device=tokens.device))
        logits = F.linear(selected,self.prediction_weight)
        return (logits,tuple(thoughts)) if return_states else logits


def matrix_work(config, length, stride, steps, batch=1, *, ordinary_full=False):
    validate(stride,steps,'latent')
    if type(length) is not int or not 0 < length <= config.max_seq or type(batch) is not int or batch < 1:
        raise ValueError('Positive bounded source length and batch required')
    if ordinary_full and steps:
        raise ValueError('Full ordinary path has no workspace')
    d, kv, layers = config.dim, config.kv_heads*(config.dim//config.heads),config.layers
    unit = layers*(4*d*d+4*d*kv+6*d*config.ffn_dim)
    projection = attention = stored = calls = 0
    for left in range(0,length,length if ordinary_full else stride):
        right = min(left+(length if ordinary_full else stride),length)
        n = right-left
        projection += n*unit
        attention += 4*layers*d*n*(stored+n)
        stored += n
        calls += 1
        if right % stride == 0:
            for _ in range(steps):
                projection += unit
                attention += 4*layers*d*(stored+1)
                stored += 1
                calls += 1
    head = 2*length*d*config.vocab_size
    forward = batch*(projection+attention+head)
    return dict(forward_matmul_flops=forward,training_matmul_flops=3*forward,
                projection_forward_flops=batch*projection,attention_forward_flops=batch*attention,
                head_forward_flops=batch*head,stored_positions=stored,sequential_body_calls=calls,
                scope='Exact dispatched dense rectangles and final real-target head; FMA=2 and backward=2x forward. Chunked ordinary must be reported separately because it also avoids some masked attention rectangles. Serial calls, elementwise operations, communication and actual latency are not modeled.')
