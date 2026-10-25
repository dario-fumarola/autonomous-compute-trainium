import torch
from compute_trainium.natural_chain_mixed import MixedConclusionObjective,coefficients,visible_stage,optimizer_schedule

class DummyModel(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.randn(7,11))
    def forward(self,inputs,state,total_steps,backward_cap):
        assert total_steps==backward_cap
        return self.weight[inputs]


def test_mixed_ntp_task_gradient_and_global_counts_match_independent_cross_entropy():
    torch.manual_seed(853);model=DummyModel();objective=MixedConclusionObjective(model)
    ntp=torch.randint(7,(2,5));target=torch.randint(11,(2,5));mask=torch.ones_like(ntp,dtype=torch.bool);stop=torch.zeros_like(mask)
    inputs=torch.randint(7,(2,3));answers=torch.randint(11,(2,3));answer_mask=torch.tensor([[0,1,0],[1,1,0]],dtype=torch.bool);eos=torch.tensor([[0,0,1],[0,0,1]],dtype=torch.bool)
    # Small tensors represent two contributions to a fixed global128-task batch.
    left,_=objective(ntp,target,mask,stop,None,coefficients(10,3,kind='ntp',task_weight=.3,world=1),1)
    right,_=objective(inputs,answers,answer_mask,eos,None,coefficients(10,3,kind='task',task_weight=.3,world=1),2)
    g=torch.autograd.grad(left+right,model.weight)[0]
    a=torch.nn.functional.cross_entropy(model.weight[ntp].flatten(0,1),target.flatten())
    b=torch.nn.functional.cross_entropy(model.weight[inputs].flatten(0,1),answers.flatten(),reduction='none').view_as(inputs)
    expected=a+.3*(b*answer_mask).sum()/3+.03*(b*eos).sum()/128
    torch.testing.assert_close(g,torch.autograd.grad(expected,model.weight)[0],rtol=1e-5,atol=1e-7)


def test_each_endpoint_completes_its_own_cooldown_and_curriculum():
    class Optimizer: param_groups=[dict(lr=1.,kind='muon',base_lr=1.)]
    opt=Optimizer()
    for length in [200,257]:
        assert visible_stage(0,length)==0 and visible_stage(length-1,length)==2
        assert optimizer_schedule(opt,length-1,length)==.0001
        assert optimizer_schedule(opt,9,length)==.001
