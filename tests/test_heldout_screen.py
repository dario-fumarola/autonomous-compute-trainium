"""Numerical duplicate-screen fixtures, never research examples."""
import numpy as np
import pytest

from compute_trainium.corpus_audit import NearDuplicateIndex
from compute_trainium.heldout_screen import CandidateIndex,PriorNeighborhoods,locate_hashes,panel_role


def test_mapped_identity_lookup_handles_edges_and_repeated_generations():
    arrays=[np.asarray(['1'*64,'3'*64],dtype='S64'),np.asarray(['5'*64],dtype='S64')]
    generation,position=locate_hashes(arrays,['0'*64,'1'*64,'3'*64,'4'*64,'5'*64,'9'*64])
    assert generation.tolist()==[-1,0,0,-1,1,-1]
    assert position[generation>=0].tolist()==[0,1,0]
    with pytest.raises(ValueError):locate_hashes([arrays[0],arrays[0]],['1'*64])


def test_inverted_screen_matches_full_prior_index_and_deduplicates_raw_rows():
    signatures=np.arange(96,dtype=np.uint64).reshape(3,32)
    sketches=[np.arange(256,dtype=np.uint64),np.arange(200,456,dtype=np.uint64),np.arange(600,856,dtype=np.uint64)]
    prior_signatures=[signatures[0].copy(),signatures[1].copy(),signatures[1].copy()]
    prior_sketches=[sketches[0],np.arange(500,756,dtype=np.uint64),np.arange(700,956,dtype=np.uint64)]
    normal=NearDuplicateIndex();inverse=CandidateIndex(signatures,sketches);summary=PriorNeighborhoods(3)
    for old,(signature,sketch) in enumerate(zip(prior_signatures,prior_sketches)):
        normal.add(signature,sketch)
        for candidate,near in inverse.compare(signature,sketch):
            summary.observe(candidate,(0,old),near);summary.observe(candidate,(0,old),near)
    assert [summary.reason(i) for i in range(3)]==[normal.check(s,k) for s,k in zip(signatures,sketches)]
    assert len(summary.neighbors[1])==2


def test_crowding_includes_old_and_new_neighbors_and_is_monotonic():
    summary=PriorNeighborhoods(1)
    for i in range(256):summary.observe(0,(0,i),False)
    assert summary.reason(0) is None
    assert summary.reason(0,additional_count=1)=='crowded_near_duplicate_bucket'
    summary.observe(0,(1,0),False);summary.observe(0,(1,1),True)
    assert len(summary.neighbors[0])==257
    assert summary.reason(0)=='crowded_near_duplicate_bucket'
    assert panel_role('c'*64)==panel_role('c'*64)
    assert panel_role('c'*64) in ['tune','curve','test']
