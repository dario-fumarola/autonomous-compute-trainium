import torch
from compute_trainium.natural_chain_objective import answer_parts,task_coefficients,task_depth,bridge_visible


def test_masked_global_objective_is_invariant_to_microbatch_partition():
    torch.manual_seed(103);logits=torch.randn(3,7,11,requires_grad=True);targets=torch.randint(11,(3,7));answer=torch.zeros(3,7,dtype=torch.bool);stop=answer.clone()
    for row,length in enumerate([1,2,3]):answer[row,2:2+length]=True;stop[row,2+length]=True
    coeff=task_coefficients(6,3);full=(answer_parts(logits,targets,answer,stop)*coeff).sum();g=torch.autograd.grad(full,logits)[0]
    split=sum((answer_parts(logits[i:i+1],targets[i:i+1],answer[i:i+1],stop[i:i+1])*coeff).sum() for i in range(3));g2=torch.autograd.grad(split,logits)[0]
    torch.testing.assert_close(g,g2);assert torch.all(g[~(answer|stop)]==0)


def test_supported_terminal_depths_and_hidden_bridge_do_not_depend_on_targets():
    assert [task_depth('recurrent',i) for i in range(6)]==[4,1,2,4,1,2]
    assert all(task_depth('one',i)==1 for i in range(121))
    assert [bridge_visible(3,i) for i in [0,39,40,79,80,120]]==[3,3,1,1,0,0]
