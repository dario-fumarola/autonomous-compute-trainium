import torch
from compute_trainium.model import ModelConfig
from compute_trainium.probability_workspace import ProbabilityWorkspace
from compute_trainium.probability_intervention import ProposalReadout


def test_identity_permutation_matches_generated_computation():
    torch.manual_seed(9);c=ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=23,max_seq=16)
    m=ProbabilityWorkspace(c).double();x=torch.randint(23,(2,9))
    torch.testing.assert_close(ProposalReadout(m,2,permutation=torch.arange(23))(x),m(x,2),rtol=0,atol=0)


def test_changed_proposal_content_recomputes_causally():
    torch.manual_seed(9);c=ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=23,max_seq=16)
    m=ProbabilityWorkspace(c).double();x=torch.randint(23,(2,9));changed=x.clone();changed[:,5:]=(changed[:,5:]+1)%23
    reader=ProposalReadout(m,2,permutation=torch.arange(22,-1,-1))
    assert not torch.equal(reader(x),m(x,2))
    torch.testing.assert_close(reader(x)[:,:5],reader(changed)[:,:5],rtol=0,atol=0)
