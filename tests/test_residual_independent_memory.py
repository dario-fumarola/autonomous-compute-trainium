import copy
import pytest
import torch
from compute_trainium.residual_workspace import ResidualWorkspace
from compute_trainium.residual_memory_intervention import read_with_memory
from compute_trainium.residual_independent_memory import independent_read,source_group_derangement,fixed_evaluation_donors


def test_fast_donor_path_matches_independent_reference_outputs_and_all_gradients():
    torch.manual_seed(7297);m=ResidualWorkspace(16,8,mode='one').double()
    with torch.no_grad():m.output.weight.normal_(std=.03)
    other=copy.deepcopy(m);x=torch.randn(2,9,16,dtype=torch.float64);d=torch.randn_like(x);a=torch.tensor([0,3,7]);target=torch.randn(2,3,16,dtype=torch.float64)
    actual=independent_read(m,x,a,d);reference=read_with_memory(other,x,a,donor=d)
    torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-12)
    (actual*target).sum().backward();(reference*target).sum().backward()
    for name,p in m.named_parameters():torch.testing.assert_close(p.grad,dict(other.named_parameters())[name].grad,rtol=1e-10,atol=1e-12)


def test_original_query_only_and_causal_donor_gradient():
    torch.manual_seed(7298);m=ResidualWorkspace(16,8,mode='one').double()
    with torch.no_grad():m.output.weight.normal_(std=.03)
    x=torch.randn(1,9,16,dtype=torch.float64,requires_grad=True);d=torch.randn_like(x,requires_grad=True);a=torch.tensor([3]);independent_read(m,x,a,d).square().mean().backward()
    assert x.grad[:,[0,1,2,4,5,6,7,8]].abs().max()==0
    torch.testing.assert_close(d.grad[:,3:],torch.zeros_like(d.grad[:,3:]),rtol=0,atol=1e-14)
    assert d.grad[:,:3].norm()>0
    torch.testing.assert_close(independent_read(m,x,a,x),m(x,a),rtol=1e-12,atol=1e-12)


def test_permutation_balances_memory_exposure_and_keeps_groups_separate():
    domains=[0]*6+[1]*4;groups=['a','a','a','b','b','c','d','d','e','f'];p=source_group_derangement(domains,groups)
    assert sorted(p)==list(range(10))
    assert all(groups[i]!=groups[j] and domains[i]==domains[j] for i,j in enumerate(p))
    with pytest.raises(ValueError):source_group_derangement([0]*4,['a','a','a','b'])
    donors=fixed_evaluation_donors(domains,groups,[0,1],['z','z'])
    for domain,row in enumerate(donors):assert domains[row[0]]==domains[row[1]]==domain and groups[row[0]]!=groups[row[1]]
