import copy

import pytest
import torch

from scripts.audit_scale_model import moments_from_shards


def rows():
    schema=dict(groups=[['lookup','norm'],['matrix']],owners=[0],placements={'lookup':'S(0)','norm':'R','matrix':'S(0)'})
    result=[]
    for rank in range(2):
        result.append(dict(optimizer=dict(schema=schema,adamw=dict(param_groups=[dict(params=[0,1])],state={
            0:dict(step=torch.tensor(2.),exp_avg=torch.full((1,2),rank+1.),exp_avg_sq=torch.full((1,2),rank+3.)),
            1:dict(step=torch.tensor(2.),exp_avg=torch.tensor([.2]),exp_avg_sq=torch.tensor([.3]))}),
            muon={0:dict(m=torch.arange(4.).reshape(1,2,2),v=torch.ones(1,2,1))} if rank==0 else {})))
    return result


def test_independent_moment_assembly_checks_replicas_owners_and_row_order():
    source=rows();out=moments_from_shards(source)
    assert torch.equal(out['lookup']['m'],torch.tensor([[1.,1.],[2.,2.]]))
    assert torch.equal(out['matrix']['m'],torch.arange(4.).reshape(2,2))
    bad=copy.deepcopy(source);bad[1]['optimizer']['adamw']['state'][1]['exp_avg'].add_(1)
    with pytest.raises(AssertionError,match='Replicated AdamW'):moments_from_shards(bad)
    bad=copy.deepcopy(source);bad[1]['optimizer']['muon']=copy.deepcopy(bad[0]['optimizer']['muon'])
    with pytest.raises(AssertionError,match='outside its owner'):moments_from_shards(bad)
