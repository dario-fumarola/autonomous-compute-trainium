"""Numerical fixtures;exercise the unchanged evaluator's actual depth API."""
import os
from pathlib import Path
import pytest
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_evaluation import prediction_state
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_mixing_control import MixingOnlyFutureSpanCandidate
from compute_trainium.mixing_prediction_reference_v2 import make_reference


def test_prediction_prefix_tail_convention_equals_total_depth_without_gradients(monkeypatch):
    upstream=os.environ.get('PARCAE_REFERENCE_ROOT')
    if not upstream or not (Path(upstream)/'parcae_lm').is_dir():
        pytest.skip('Set PARCAE_REFERENCE_ROOT to the separately pinned upstream checkout')
    monkeypatch.syspath_prepend(upstream)
    torch.manual_seed(3931)
    c=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=41,max_seq=16)
    model=MixingOnlyFutureSpanCandidate(c,FutureSpanConfig(context=16,horizon=4,stride=2),
        initialize=False,fp16_body=False).eval()
    ref=make_reference(c,True).eval();ref.load_state_dict(prediction_state(model.state_dict()),strict=True)
    tokens=torch.arange(16)[None]%41;noise=torch.randn(1,16,32)
    ref.initialize_state=lambda _:noise
    with sdpa_kernel(SDPBackend.MATH),torch.inference_mode():
        for depth in [1,2,3]:
            actual=ref(tokens,return_logits=True,num_steps_pair=torch.tensor([depth-1,1]))['logits']
            equivalent=ref(tokens,return_logits=True,num_steps_pair=torch.tensor([0,depth]))['logits']
            trained=model(tokens,noise,total_steps=depth,backward_cap=depth)[0]
            torch.testing.assert_close(actual,equivalent,rtol=0,atol=0)
            torch.testing.assert_close(actual,trained,rtol=0,atol=0)
    with pytest.raises(ValueError,match='Prediction-only'):
        ref(tokens,return_logits=True,num_steps_pair=torch.tensor([0,2]))
