import pytest
import torch
from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig,rotary_tensors
from compute_trainium.sparse_workspace_candidate import SparseWorkspaceCandidate
from compute_trainium.sparse_workspace_training import ANCHORS,pack_view,sparse_objective,sparse_matrix_work
from test_workspace_matrix_count import MatrixCounter


@pytest.mark.parametrize('mode',['ordinary','latent','constant'])
def test_actual_matrix_dispatch_matches_body_recompute_and_selected_head(mode):
    c=OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=64,pad_token_id=63)
    torch.manual_seed(8);model=SparseWorkspaceCandidate(c,activation_checkpointing=True)
    full=torch.arange(160).reshape(1,160)%62
    tokens,targets=pack_view(dict(prefix=full[:,:128],continuation=full[:,128:]),c.pad_token_id)
    assert torch.equal(targets,full[:,1:])
    assert int(targets[0,127])==int(full[0,128])
    counter=MatrixCounter()
    with counter:
        loss,raw,anchor=sparse_objective(model,tokens,targets,*rotary_tensors(c,160),torch.tensor(ANCHORS),mode=mode,pause_id=62)
        loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw) and torch.isfinite(anchor)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert counter.operations==sparse_matrix_work(c,mode)
