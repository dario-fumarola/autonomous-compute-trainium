import sys
from types import ModuleType,SimpleNamespace

import pytest
import torch

from compute_trainium.reference_attention_mask_fixture import explicit_reference_mask


def test_mask_reaches_attention_and_override_restores_after_failure(monkeypatch):
    original=object();flash=SimpleNamespace(flash_attn_func=original)
    modules=ModuleType('parcae_lm.modules');modules.mixer=SimpleNamespace(flash_attn=flash)
    monkeypatch.setitem(sys.modules,'parcae_lm.modules',modules)
    mask=torch.eye(3,dtype=torch.bool);q=torch.ones(1,3,1,2);v=torch.arange(6.).reshape(1,3,1,2)
    with pytest.raises(RuntimeError):
        with explicit_reference_mask(mask):
            assert torch.equal(flash.flash_attn_func(q,q,v,causal=True),v)
            raise RuntimeError('Fixture failure')
    assert flash.flash_attn_func is original
