import torch
from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig
from compute_trainium.future_span_evaluation import window_state,prediction_state
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_stages_candidate import ParcaeStagesCandidate


def test_noise_and_prediction_prefix_are_invariant_to_future_tokens_and_prefix_length():
    torch.manual_seed(827)
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64,max_seq=32)
    short=window_state(config,3,999,8);long=window_state(config,3,999,16)
    assert torch.equal(short,long[:,:8])
    model=ParcaeStagesCandidate(config)
    tokens=torch.arange(16)[None]
    for depth in [1,2,3]:
        a=model(tokens[:,:8],short,total_steps=depth,backward_cap=depth)
        b=model(tokens,long,total_steps=depth,backward_cap=depth)
        torch.testing.assert_close(a,b[:,:8],atol=1e-6,rtol=1e-6)
    span=FutureSpanConfig(context=16,horizon=4,stride=3)
    training=FutureSpanCandidate(config,span,fp16_body=False)
    model.load_state_dict(prediction_state(training.state_dict()),strict=True)
    expected,_=training(tokens,long,total_steps=2,backward_cap=2)
    torch.testing.assert_close(model(tokens,long,total_steps=2,backward_cap=2),expected,rtol=0,atol=0)
