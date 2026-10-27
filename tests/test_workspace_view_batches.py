import numpy as np
import pytest
import torch
from compute_trainium.workspace_view_batches import WorkspaceViewBatches


def test_shared_source_order_context_split_and_no_unregistered_repetition(tmp_path):
    values=np.arange(3*160,dtype=np.uint32).reshape(3,160);np.save(tmp_path/'train-tokens.npy',values)
    np.save(tmp_path/'train-order.npy',np.array([2,0,1],dtype=np.uint32));np.save(tmp_path/'train-domains.npy',np.array([0,1,2],dtype=np.uint8))
    data=WorkspaceViewBatches(tmp_path,'train');row=data.at(0)
    assert row['source_row']==2 and row['domain']==2
    assert torch.equal(row['prefix'],torch.tensor(values[2,:128].astype(np.int64)).reshape(1,-1))
    assert row['continuation'].shape==(1,32)
    with pytest.raises(IndexError):data.at(3)
