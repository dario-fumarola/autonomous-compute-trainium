"""Read-only shared source arrays;workers never parse a full Python corpus."""
import numpy as np
import torch


class WorkspaceViewBatches:
    def __init__(self,root,split):
        self.tokens=np.load(root/(split+'-tokens.npy'),mmap_mode='r')
        self.order=np.load(root/(split+'-order.npy'),mmap_mode='r')
        self.domains=np.load(root/(split+'-domains.npy'),mmap_mode='r')
        if self.tokens.shape!=(len(self.order),160) or self.domains.shape!=(len(self.order),):raise ValueError('Exact source view geometry required')
    def __len__(self):return len(self.order)
    def at(self,index):
        if type(index) is not int or not 0<=index<len(self):raise IndexError('Frozen fresh-source allowance exhausted;no silent epoch wrap')
        row=int(self.order[index]);tokens=torch.from_numpy(self.tokens[row].astype(np.int64,copy=True))
        return dict(prefix=tokens[:128].reshape(1,-1),continuation=tokens[128:].reshape(1,-1),domain=int(self.domains[row]),source_row=row)
