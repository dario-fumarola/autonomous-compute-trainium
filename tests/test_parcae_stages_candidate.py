import torch

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig
from compute_trainium.parcae_stages_candidate import ParcaeStagesCandidate


def test_stages_preserve_module_transfer_checkpoint_and_gradients():
    torch.manual_seed(302)
    c = ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64,max_seq=16)
    model = ParcaeCandidate(c,initialize=False).to('cpu')
    staged = ParcaeStagesCandidate(c,initialize=False).to('cpu')
    staged.load_state_dict(model.state_dict())
    assert list(staged.state_dict()) == list(model.state_dict())
    x, state = torch.arange(8)[None], torch.randn(1,8,32)
    a = model(x,state,total_steps=4,backward_cap=2)
    b = staged(x,state,total_steps=4,backward_cap=2)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.square().sum().backward()
    b.square().sum().backward()
    for pa,pb in zip(model.parameters(),staged.parameters(),strict=True):
        torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
