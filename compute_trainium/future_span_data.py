"""Static auxiliary-count preparation; original packing/target order unchanged."""
import numpy as np


def count_future_anchors(domains, config, windows, *, chunk_windows=4096):
    """Return per-window valid anchor counts and counts by original domain.

    Domain zero marks all artificial boundaries and padding. Integer prefix
    sums test the entire inclusive interval [t,t+horizon],including t+1 even
    though that immediate-next token is excluded from auxiliary labels.
    Chunking changes neither anchors nor source order and bounds memory.
    """
    if (domains.ndim!=1 or windows<1 or chunk_windows<1
            or len(domains)<windows*config.context+1 or len(config.positions)>255):
        raise ValueError('Provisioned packed domains and bounded static anchor slots required')
    positions=np.asarray(config.positions,dtype=np.int64)
    counts=np.empty(windows,dtype=np.uint8)
    by_domain=np.zeros(3,dtype=np.int64)
    for start in range(0,windows,chunk_windows):
        stop=min(windows,start+chunk_windows)
        original=np.asarray(domains[start*config.context:stop*config.context+1])
        if original.min()<0 or original.max()>3:raise ValueError('Only boundary0 and original domains1/2/3 are defined')
        invalid=np.empty(len(original)+1,dtype=np.int32);invalid[0]=0
        np.cumsum(original==0,out=invalid[1:])
        anchors=np.arange(stop-start,dtype=np.int64)[:,None]*config.context+positions[None,:]
        valid=(invalid[anchors+config.horizon+1]-invalid[anchors])==0
        counts[start:stop]=valid.sum(-1)
        labels=original[anchors]
        for domain in range(1,4):by_domain[domain-1]+=np.count_nonzero(valid & (labels==domain))
    if int(counts.sum(dtype=np.int64))!=int(by_domain.sum()):
        raise AssertionError('Auxiliary anchors must belong to exactly one source domain')
    return counts,by_domain
