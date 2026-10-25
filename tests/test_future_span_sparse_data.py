import numpy as np
import torch

from compute_trainium.future_span import FutureSpanConfig,future_span_targets,future_span_loss_sum
from compute_trainium.future_span_sparse import sparse_targets,sparse_future_loss_sum
from compute_trainium.future_span_data import count_future_anchors
from compute_trainium.training import PackedCorpus


def test_sparse_presence_loss_and_all_gradients_match_dense_with_duplicates_and_extremes():
    config=FutureSpanConfig(context=8,horizon=4,stride=2)
    source=torch.tensor([[0,1,2,2,2,3,4,4,5],[5,4,4,3,3,2,2,1,0]])
    real=torch.ones_like(source,dtype=torch.bool);real[1,2]=False
    labels=future_span_targets(source,real,config,16);sparse=sparse_targets(labels)
    for scale in [1,100]:
        z=(torch.randn(2,3,16)*scale).requires_grad_()
        original=future_span_loss_sum(z,labels,config)
        replacement=sparse_future_loss_sum(z,sparse.token_ids,sparse.distinct,sparse.valid,config.horizon)
        torch.testing.assert_close(original,replacement,atol=5e-5,rtol=2e-6)
        a=torch.autograd.grad(original,z,retain_graph=True)[0]
        b=torch.autograd.grad(replacement,z)[0]
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    assert (sparse.distinct.sum(-1)<config.horizon-1).any()


def test_chunked_future_counts_equal_independent_per_window_boundary_masks_without_ntp_loss():
    documents=[{'id':str(i),'domain':d,'split':'train','tokens':list(range(3,3+n))}
               for i,(d,n) in enumerate([('web',2),('web',19),('code',31),('math',7),('math',13)])]
    packed=PackedCorpus(documents,16)
    config=FutureSpanConfig(context=16,horizon=4,stride=3)
    expected=[];domain_counts=np.zeros(3,dtype=np.int64)
    for window in range(packed.windows):
        source=packed.tokens[window*16:window*16+17].long()[None]
        domains=packed.domains[window*16:window*16+17][None]
        targets=future_span_targets(source,domains>0,config,64)
        expected.append(int(targets.valid.sum()))
        for domain in range(1,4):
            domain_counts[domain-1]+=int(((domains[:,targets.positions]==domain)&targets.valid).sum())
    for chunk in [1,2,4096]:
        counts,by_domain=count_future_anchors(packed.domains.numpy(),config,packed.windows,chunk_windows=chunk)
        np.testing.assert_array_equal(counts,expected)
        np.testing.assert_array_equal(by_domain,domain_counts)
    all_ntp=sum(int(packed.batch([window],0)[0].target_mask.sum()) for window in range(packed.windows))
    assert all_ntp==sum(len(d['tokens']) for d in documents)==packed.real_targets
