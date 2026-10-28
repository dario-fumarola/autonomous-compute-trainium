import torch
from compute_trainium.residual_workspace import ResidualWorkspace,unit
from compute_trainium.residual_memory_intervention import read_with_memory,choose_donors


def model():
    torch.manual_seed(7291);m=ResidualWorkspace(16,8,mode='one').double()
    with torch.no_grad():m.output.weight.normal_(std=.03)
    return m


def test_identity_current_slot_and_independent_query_recomputation():
    m=model();x=torch.randn(2,9,16,dtype=torch.float64);d=torch.randn_like(x);a=torch.tensor([2,6])
    torch.testing.assert_close(read_with_memory(m,x,a),m(x,a),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(read_with_memory(m,x,a,donor=x),m(x,a),rtol=1e-12,atol=1e-12)
    actual=read_with_memory(m,x,a,donor=d)
    for j,anchor in enumerate(a):
        mixed=x.clone();mixed[:,:anchor]=d[:,:anchor]
        torch.testing.assert_close(actual[:,j:j+1],m(mixed,anchor.reshape(1)),rtol=1e-12,atol=1e-12)
        changed=d.clone();changed[:,anchor:]+=100
        torch.testing.assert_close(read_with_memory(m,x,anchor.reshape(1),donor=d),read_with_memory(m,x,anchor.reshape(1),donor=changed),rtol=0,atol=0)


def test_self_only_invariant_to_other_positions_and_donor_changes_content():
    m=model();x=torch.randn(2,9,16,dtype=torch.float64);a=torch.tensor([6]);y=x.clone();y[:,:6]+=30;y[:,7:]-=20
    assert torch.equal(read_with_memory(m,x,a,self_only=True),read_with_memory(m,y,a,self_only=True))
    assert not torch.allclose(read_with_memory(m,x,a,donor=y),m(x,a),rtol=1e-8,atol=1e-8)
    # A joint prefix K/V permutation is an invariance,not a useful corruption.
    perm=x.clone();perm[:,:6]=x[:,torch.tensor([4,2,5,0,3,1])]
    torch.testing.assert_close(m(x,a),m(perm,a),rtol=1e-12,atol=1e-12)


def test_donor_selection_uses_distinct_source_groups_and_length():
    r=[dict(stratum='code',group=g,length=n) for g,n in [('a',4),('a',8),('b',8),('c',6),('d',3)]]
    chosen,excluded=choose_donors(r)
    assert '0' in chosen and 1 in excluded
    for i,donors in chosen.items():
        assert len({r[j]['group'] for j in donors}|{r[int(i)]['group']})==3
        assert all(r[j]['length']>=r[int(i)]['length'] for j in donors)
