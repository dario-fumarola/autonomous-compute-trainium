import copy
import torch
from compute_trainium.residual_workspace import ResidualWorkspace


def test_exact_identity_and_frozen_base_then_branch_learns():
 torch.manual_seed(7281);h=torch.randn(2,9,16);head=torch.randn(31,16);a=torch.tensor([2,5,8]);target=torch.randint(31,(2,3))
 for mode in ['linear','one','two']:
  m=ResidualWorkspace(16,8,mode=mode);opt=torch.optim.AdamW(m.parameters(),lr=.003);old_h=h.clone();old_head=head.clone()
  assert torch.equal(m(h,a),h[:,a])
  for step in range(2):
   opt.zero_grad();loss=torch.nn.functional.cross_entropy((m(h,a)@head.T).reshape(-1,31),target.flatten());loss.backward()
   assert m.output.weight.grad.abs().sum()>0
   if step==1:assert m.input.weight.grad.abs().sum()>0
   opt.step()
  assert torch.equal(m(h,a,disabled=True),h[:,a]);assert torch.equal(h,old_h) and torch.equal(head,old_head)


def test_future_features_and_other_queries_cannot_leak_to_prediction():
 torch.manual_seed(7282);m=ResidualWorkspace(16,8);torch.nn.init.normal_(m.output.weight,std=.01);h=torch.randn(1,9,16);a=torch.tensor([3,7]);baseline=m(h,a);changed=h.clone();changed[:,4:]=torch.randn_like(changed[:,4:])*100
 torch.testing.assert_close(m(changed,a)[:,0],baseline[:,0],rtol=0,atol=0)
 torch.testing.assert_close(m(h,a[:1]),baseline[:,:1],rtol=1e-5,atol=1e-6)


def test_recurrence_has_same_parameters_and_only_declared_depth_differs():
 torch.manual_seed(7283);one=ResidualWorkspace(16,8,mode='one');torch.manual_seed(7283);two=ResidualWorkspace(16,8,mode='two')
 assert one.state_dict().keys()==two.state_dict().keys()
 assert all(torch.equal(v,two.state_dict()[k]) for k,v in one.state_dict().items())
 torch.nn.init.normal_(one.output.weight,std=.01);two.load_state_dict(one.state_dict());h=torch.randn(1,9,16);a=torch.tensor([2,8]);assert not torch.equal(one(h,a),two(h,a))


def test_gradient_matches_finite_difference_through_repeated_cell():
 torch.manual_seed(7284);m=ResidualWorkspace(8,4).double();torch.nn.init.normal_(m.output.weight,std=.01);h=torch.randn(1,5,8,dtype=torch.double);a=torch.tensor([3]);m(h,a).square().sum().backward();p=m.cell.q.weight;actual=float(p.grad[0,0]);original=float(p[0,0].detach());eps=1e-5
 with torch.no_grad():
  p[0,0]=original+eps;plus=float(m(h,a).square().sum());p[0,0]=original-eps;minus=float(m(h,a).square().sum());p[0,0]=original
 assert abs(actual-(plus-minus)/(2*eps))<1e-7
