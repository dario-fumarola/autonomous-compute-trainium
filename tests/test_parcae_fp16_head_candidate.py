import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_fp16_candidate import ParcaeFP16Candidate
from compute_trainium.parcae_fp16_head_candidate import ParcaeFP16HeadCandidate


def test_head_precision_preserves_master_tying_inventory_and_scaled_gradients():
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64,max_seq=16)
    reference=ParcaeFP16Candidate(config);candidate=ParcaeFP16HeadCandidate(config,initialize=False)
    candidate.load_state_dict(reference.state_dict())
    assert dict(candidate.named_parameters()).keys()==dict(reference.named_parameters()).keys()
    assert candidate.lm_head.weight is candidate.transformer.wte.weight
    assert all(p.dtype==torch.float32 for p in candidate.parameters())
    tokens=torch.arange(16)[None];state=torch.randn(1,16,32)*.02
    before={n:t.clone() for n,t in candidate.state_dict().items()}
    with sdpa_kernel(SDPBackend.MATH):
        logits=candidate(tokens,state,total_steps=2,backward_cap=2)
        assert logits.dtype==torch.float32 and logits.shape==(1,16,64)
        (logits.square().mean()*8192).backward()
    assert all(p.grad is not None and p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all()
               for p in candidate.parameters())
    assert all(torch.equal(before[n],t) for n,t in candidate.state_dict().items())
