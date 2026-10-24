import torch

from compute_trainium.parcae_optimizer_precise import ParcaePreciseOptimizer
from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer
from scripts.qualify_native import digest


class Inventory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer=torch.nn.Module()
        self.transformer.h=torch.nn.ModuleList([torch.nn.Linear(8,12,bias=False),torch.nn.Linear(12,8,bias=False)])
        self.transformer.wte=torch.nn.Embedding(19,8)


def test_changing_runtime_scalars_preserve_every_parameter_and_moment():
    torch.manual_seed(18);a=Inventory();b=Inventory();b.load_state_dict(a.state_dict())
    left=ParcaePreciseOptimizer(a,gate_policy='release_muon');right=ParcaeRuntimeScalarOptimizer(b,gate_policy='release_muon')
    for update,(lr,momentum,decay) in enumerate([(.0001,.855,.2),(.008,.95,.12),(.0023,.91,.019)]):
        for x,y in zip(a.parameters(),b.parameters()):x.grad=torch.randn_like(x);y.grad=x.grad.clone()
        for optimizer in [left,right]:
            for group in optimizer.param_groups:
                group['lr']=lr*group.get('base_lr',1.)
                if group['kind']=='muon':group['momentum']=momentum;group['weight_decay']=decay
            optimizer.step()
        assert digest({'model':a.state_dict(),'optimizer':left.state_dict()})==digest({'model':b.state_dict(),'optimizer':right.state_dict()})
    c=Inventory();c.load_state_dict(b.state_dict());recovered=ParcaeRuntimeScalarOptimizer(c,gate_policy='release_muon')
    recovered.load_state_dict(right.state_dict())
    assert digest(recovered.state_dict())==digest(right.state_dict())
