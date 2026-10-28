"""Count actual model-level matrix dispatches independently of the ledger."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig,rotary_tensors
from compute_trainium.workspace_discovery import forward_matrix_work


class MatrixCounter(TorchDispatchMode):
    def __init__(self):super().__init__();self.operations=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if func in [torch.ops.aten.mm.default,torch.ops.aten.bmm.default]:
            left,right=args[:2];self.operations+=2*left.numel()*right.shape[-1]
        return func(*args,**(kwargs or {}))


@pytest.mark.parametrize('checkpointing',[False,True])
def test_postnorm_candidate_executes_declared_recomputation_and_selected_head(checkpointing):
    c=OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=64,pad_token_id=63)
    model=WorkspaceOlmoCandidate(c,activation_checkpointing=checkpointing)
    tokens=torch.arange(32).reshape(1,32);positions=torch.arange(7);counter=MatrixCounter()
    with counter:model(tokens,*rotary_tensors(c,32),positions=positions).square().mean().backward()
    forward=forward_matrix_work(c,batch=1,positions=32,projected=7)
    # Postnorm needs the final MLP output during recomputation;no projection
    # is skipped by nonreentrant early stopping.All shapes are software fixtures.
    assert counter.operations==(4 if checkpointing else 3)*forward['body']+3*forward['head']
