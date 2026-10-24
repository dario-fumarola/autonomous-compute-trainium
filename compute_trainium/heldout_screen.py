"""Small-candidate duplicate screening without storing the entire prior corpus.

The matching/rejection rule is identical to the existing8x4 minhash candidate
rule, bottom256 verification and crowded-neighborhood cutoff. Only direction
of indexing changes: prior raw documents stream against a fixed small panel.
"""
import hashlib

import numpy as np

from .corpus_audit import bottom_k_jaccard


def panel_role(group_hash):
    bucket=int(hashlib.sha256(('confirmation-reserve-v1:'+group_hash).encode()).hexdigest()[:16],16)%3
    return ['tune','curve','test'][bucket]


def locate_hashes(arrays, identities):
    keys=np.asarray(identities,dtype='S64');generation=np.full(len(keys),-1,dtype=np.int16)
    positions=np.zeros(len(keys),dtype=np.int64)
    for number,array in enumerate(arrays):
        offsets=np.searchsorted(array,keys);valid=offsets<len(array);matched=np.zeros(len(keys),dtype=bool)
        matched[valid]=array[offsets[valid]]==keys[valid]
        if np.any(generation[matched]>=0):raise ValueError('Accepted identity occurs in multiple generations')
        generation[matched]=number;positions[matched]=offsets[matched]
    return generation,positions


class CandidateIndex:
    def __init__(self,signatures,sketches):
        self.sketches=sketches;self.bands={}
        for index,signature in enumerate(signatures):
            for key in self.keys(signature):self.bands.setdefault(key,[]).append(index)

    @staticmethod
    def keys(signature):return [(i,signature[4*i:4*i+4].tobytes()) for i in range(8)]

    def matching(self,signature):
        matched=set()
        for key in self.keys(signature):matched.update(self.bands.get(key,()))
        return matched

    def compare(self,signature,sketch):
        return [(index,bottom_k_jaccard(sketch,self.sketches[index])>=.8) for index in sorted(self.matching(signature))]


class PriorNeighborhoods:
    """Deduplicate raw appearances by the accepted identity's stable index."""
    def __init__(self,count,maximum=256):
        self.neighbors=[set() for _ in range(count)];self.near=np.zeros(count,dtype=bool);self.maximum=maximum

    def observe(self,candidate,old_identity,is_near):
        if len(self.neighbors[candidate])<=self.maximum:self.neighbors[candidate].add(tuple(old_identity))
        self.near[candidate]|=is_near

    def reason(self,candidate,additional_count=0):
        if len(self.neighbors[candidate])+additional_count>self.maximum:return 'crowded_near_duplicate_bucket'
        return 'near_duplicate' if self.near[candidate] else None
