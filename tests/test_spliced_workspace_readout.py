import pytest
import torch
import torch.nn.functional as F
from compute_trainium.spliced_workspace_readout import readout_grid,choose_readout


def test_every_readout_matches_independent_cross_entropy_and_endpoints():
    torch.manual_seed(5611);raw=torch.randn(2,32,17,dtype=torch.float64);worked=torch.randn(2,4,17,dtype=torch.float64);labels=torch.randint(0,17,(2,32));selected=torch.tensor([0,8,16,24]);alphas=[0,.05,.2,.5];temps=[1,.9,1.1]
    result=readout_grid(raw,worked,labels,selected,alphas=alphas,temperatures=temps)
    for a,alpha in enumerate(alphas):
        for t,temp in enumerate(temps):
            logits=raw.clone();logits[:,selected]=raw[:,selected]+2*alpha*(worked-raw[:,selected]);expected=F.cross_entropy((logits/temp).reshape(-1,17),labels.reshape(-1))
            torch.testing.assert_close(result['policy32'][a,t],expected,rtol=1e-12,atol=1e-12)
    assert torch.equal(result['policy32'][0],result['raw32'])
    expected=F.cross_entropy(worked.reshape(-1,17),labels[:,selected].reshape(-1))
    torch.testing.assert_close(result['worked_anchor'][-1,0],expected)


def test_global_selection_uses_train_only_and_respects_ordinary_compute():
    rows=[dict(domain=d,raw32=[2.,2.1],policy32=[[2.,2.1],[1.9,2.05]]) for d in range(3)]
    kw=dict(alphas=[0,.5],temperatures=[1.,.9]);best=choose_readout(rows,**kw,split='train');assert best['alpha']==.5 and best['temperature']==1.
    raw=choose_readout(rows,**kw,ordinary=True,split='train');assert raw['alpha']==0 and raw['temperature']==1.
    with pytest.raises(ValueError):choose_readout(rows,**kw,split='tune')
    # Exactly equal choices prefer zero extra computation and unit temperature.
    tied=[dict(domain=d,raw32=[2.,2.],policy32=[[2.,2.],[2.,2.]]) for d in range(3)]
    assert choose_readout(tied,**kw,split='train')['alpha']==0
