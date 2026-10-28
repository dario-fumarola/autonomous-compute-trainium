"""Read-only natural-source packing for the registered replay screen."""
import json
from pathlib import Path
import numpy as np
import torch


class ProbabilitySources:
    def __init__(self, root):
        self.root=Path(root);self.report=json.loads((self.root/'report.json').read_text())
        if self.report['status']!='PASS_PROBABILITY_WORKSPACE_NATURAL_SOURCES':raise ValueError('Source acceptance required')
        self.tokens=np.load(self.root/'train-tokens.npy',mmap_mode='r')
        self.domains=np.load(self.root/'train-domains.npy',mmap_mode='r')
        self.context=self.report['context'];self.windows=self.report['windows']

    def batch(self, indices):
        if not indices or min(indices)<0 or max(indices)>=self.windows:raise ValueError('Unique source allowance exhausted')
        raw=np.stack([self.tokens[i*self.context:(i+1)*self.context+1] for i in indices]).astype(np.int64)
        mask=np.stack([self.domains[i*self.context+1:(i+1)*self.context+1]>0 for i in indices])
        return torch.from_numpy(raw[:,:-1].copy()),torch.from_numpy(raw[:,1:].copy()),torch.from_numpy(mask)

    def target_count(self,start,windows):
        if min(start,windows)<0 or start+windows>self.windows:raise ValueError('Invalid source interval')
        return int(np.count_nonzero(self.domains[start*self.context+1:(start+windows)*self.context+1]))
