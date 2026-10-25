"""Teacher-free future-token presence targets; prospective CPU candidate only.

Uses the hand-crafted target in FSP (arXiv:2510.14751v2, eqs.9–10), with
explicit constant normalization, sparse anchors and no corpus-fitted weights.
No current scientific trainer imports this module.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .parcae_fp16_candidate import ParcaeFP16Candidate


@dataclass(frozen=True)
class FutureSpanConfig:
    context: int = 1024
    horizon: int = 32
    stride: int = 16
    weight: float = .1

    def __post_init__(self):
        if (any(type(x) is not int for x in [self.context,self.horizon,self.stride])
                or not 2 <= self.horizon <= self.context or self.stride < 1
                or not math.isfinite(self.weight) or self.weight < 0):
            raise ValueError('Fixed integer context/horizon/stride and nonnegative weight required')

    @property
    def positions(self):
        # A source row has context+1 entries; the backbone sees its first context.
        return tuple(range(0,self.context-self.horizon+1,self.stride))


@dataclass
class FutureSpanTargets:
    positions: torch.Tensor
    token_ids: torch.Tensor
    valid: torch.Tensor


def future_span_targets(source, real_mask, config, vocab_size):
    """Host-side labels only, never inputs to either prediction head.

    At input position t, predict presence in source[t+2:t+horizon+1]. The
    immediate next token is excluded. Require every source position from t
    through t+horizon to be real: artificial boundaries/padding invalidate the
    entire anchor. No shortened spans, crossing documents, wrapping or extra
    source reads. Every ordinary next-token label remains available separately.
    """
    if (source.device.type!='cpu' or real_mask.device.type!='cpu'
            or source.dtype!=torch.long or real_mask.dtype!=torch.bool
            or source.ndim!=2 or source.shape[1]!=config.context+1
            or real_mask.shape!=source.shape or vocab_size<=config.horizon
            or source.numel()==0 or source.min()<0 or source.max()>=vocab_size):
        raise ValueError('Aligned CPU source IDs and real-position mask required')
    positions=torch.tensor(config.positions,dtype=torch.long)
    indices=positions[:,None]+torch.arange(2,config.horizon+1)[None,:]
    spans=positions[:,None]+torch.arange(config.horizon+1)[None,:]
    return FutureSpanTargets(positions,source[:,indices],real_mask[:,spans].all(-1))


def future_span_loss_sum(logits, targets, config):
    """Sum over valid anchors; caller divides by GLOBAL valid-anchor count.

    Vocabulary BCE is summed and divided by horizon-1, not vocabulary size.
    Duplicates are presence (one), not multiplicity. All anchors are executed,
    including invalid ones; masking their losses does not save their compute.
    """
    if (logits.ndim!=3 or targets.token_ids.shape!=(*logits.shape[:2],config.horizon-1)
            or targets.valid.shape!=logits.shape[:2]
            or logits.shape[1]!=len(config.positions)):
        raise ValueError('Static anchor/logit/target dimensions differ')
    labels=torch.zeros_like(logits).scatter_(-1,targets.token_ids,1.)
    terms=F.binary_cross_entropy_with_logits(logits.float(),labels.float(),reduction='none')
    return (terms.sum(-1)*targets.valid).sum()/(config.horizon-1)


def normalized_objective(ntp_sum, future_sum, *, ntp_count, future_count, weight, world=1):
    """One microbatch's contribution under DDP mean reduction and accumulation.

    Counts cover the whole optimizer batch, not this rank or microbatch.
    A globally empty auxiliary batch contributes exactly zero without NaNs.
    """
    if min(ntp_count,world)<=0 or future_count<0 or not math.isfinite(weight) or weight<0:
        raise ValueError('Global real-token/anchor counts and finite weight required')
    return world*(ntp_sum/ntp_count+weight*future_sum/max(1,future_count))


class FutureSpanCandidate(ParcaeFP16Candidate):
    """Identical causal backbone plus a separate, training-only linear head.

    All four prospective cells retain/evaluate the head, including weight-zero
    controls. Its name explicitly selects the existing AdamW head group. The
    head is not tied to token logits or fed back into prediction. Exporting only
    prediction_state_dict restores the existing ordinary causal inference path.
    Full/global native gradients, optimizer recovery and throughput are unqualified.
    """
    def __init__(self, model_config, future_config, *, head_seed=104729, **kwargs):
        super().__init__(model_config,**kwargs)
        if future_config.context>model_config.max_seq or model_config.vocab_size<=future_config.horizon:
            raise ValueError('Future spans must fit the declared backbone context/vocabulary')
        self.future_config=future_config
        # Head initialization must not consume the backbone's ongoing RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.future_lm_head=nn.Linear(model_config.dim,model_config.vocab_size,bias=True)
            nn.init.normal_(self.future_lm_head.weight,std=.02)
            p=(future_config.horizon-1)/model_config.vocab_size
            nn.init.constant_(self.future_lm_head.bias,math.log(p/(1-p)))
        self.register_buffer('future_positions',torch.tensor(future_config.positions,dtype=torch.long))

    def decode(self, state, tokens):
        if tokens.shape[1]!=self.future_config.context:
            raise ValueError('Explicit fixed training context required')
        with torch.autocast(state.device.type,dtype=torch.float16,enabled=self.fp16_body):
            if self.recurrent:
                state=self.transformer.C(state)
                for j,block in enumerate(self.transformer.coda):
                    state=self._apply_block(block,state,tokens,self.config.prelude+self.config.core+j)
        with torch.autocast(state.device.type,enabled=False):
            hidden=self.transformer.ln_f(state.float())
            return (F.linear(hidden,self.lm_head.weight.float()),
                    self.future_lm_head(hidden.index_select(1,self.future_positions)))

    def prediction_state_dict(self):
        return {k:v for k,v in self.state_dict().items()
                if not k.startswith('future_lm_head.') and k!='future_positions'}
