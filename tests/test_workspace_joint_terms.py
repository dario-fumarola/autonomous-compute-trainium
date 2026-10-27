import torch
from compute_trainium.workspace_learning import (Reader,WorkspaceFormat,target_log_probs,reader_loss,policy_loss,identity_reader_context)
from compute_trainium.workspace_joint_terms import workspace_joint_terms,padded_prediction_inputs
from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig,rotary_tensors


def test_sequential_selected_padded_objective_matches_batched_loss_and_every_gradient():
    torch.manual_seed(7181)
    model=WorkspaceOlmoCandidate(OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,vocab_size=32,pad_token_id=31))
    c=torch.tensor([[1,2,3,4]]);y=torch.tensor([[5,6]]);z=torch.tensor([[7,8],[9,10],[11,12],[13,14]])
    reward=torch.tensor([-.1,.03,.07,-.04]);fmt=WorkspaceFormat((15,),(16,))
    def logits(x):return model(x,*rotary_tensors(model.config,x.shape[1]))
    reader=Reader(logits)
    raw=-target_log_probs(reader,c[:,:1],torch.cat([c[:,1:],y],1),phase='ordinary').mean()
    read=reader_loss(reader,c,y,z,format=fmt,context_builder=identity_reader_context)
    pg=policy_loss(reader,c,z,reward,format=fmt)
    expected=.5*raw+.5*read+.1*pg;expected.backward()
    gradients={n:p.grad.clone() for n,p in model.named_parameters()};model.zero_grad(set_to_none=True)
    total=0.
    for phase,context,target,weight,reduction in workspace_joint_terms(c,y,z,reward,fmt):
        inputs,positions=padded_prediction_inputs(context,target,16,31)
        value=model(inputs,*rotary_tensors(model.config,16),positions)
        loss=-torch.log_softmax(value,-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)
        loss=weight*(loss.mean() if reduction=='mean' else loss.sum())
        total+=float(loss.detach());loss.backward()
    torch.testing.assert_close(torch.tensor(total),expected.detach(),rtol=2e-6,atol=1e-6)
    for n,p in model.named_parameters():torch.testing.assert_close(p.grad,gradients[n],rtol=2e-4,atol=2e-6)
