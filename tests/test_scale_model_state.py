import copy

import pytest

from compute_trainium.scale_model_state import PURPOSE,validate_shard
from compute_trainium.owned_native_all_chips import create_all_chip_worker


def test_large_qualification_shards_reject_wrong_topology_and_scientific_relabeling():
    value=dict(purpose=PURPOSE,qualification_only=True,eligible_for_science=False,
        signature={'plan':'frozen'},rank=17,world=64,update=100,cursor=51200)
    validate_shard(value,{'plan':'frozen'},17,64,100)
    for key,other in [('purpose','science'),('eligible_for_science',True),('rank',18),('world',32),
                      ('update',101),('cursor',51000),('signature',{'plan':'changed'})]:
        bad={**value,key:other}
        with pytest.raises(ValueError):validate_shard(bad,{'plan':'frozen'},17,64,100)


def test_worker_cap_cannot_grow_without_an_explicit_supported_allowance(tmp_path):
    with pytest.raises(ValueError):create_all_chip_worker(tmp_path,'invalid-allowance',memory_gib=512)
