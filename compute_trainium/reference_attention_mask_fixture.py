"""Explicit mask adapter for a numerical rehearsal,never scientific scoring.

The pinned upstream mixer's optional mask argument is not forwarded to its
flash-attention function. Override that function only in an isolated CPU fixture
and restore it on every exit. The on-disk upstream checkout remains unchanged.
"""
from contextlib import contextmanager
from torch.nn import functional as F


@contextmanager
def explicit_reference_mask(mask):
    from parcae_lm.modules import mixer
    original=mixer.flash_attn.flash_attn_func
    def attention(q,k,v,causal=False,window_size=(-1,-1)):
        if not causal or window_size!=(-1,-1):raise ValueError('Only the frozen causal fixture is in scope')
        return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
            attn_mask=mask,enable_gqa=q.shape[2]!=k.shape[2]).transpose(1,2)
    mixer.flash_attn.flash_attn_func=attention
    try:yield
    finally:mixer.flash_attn.flash_attn_func=original
