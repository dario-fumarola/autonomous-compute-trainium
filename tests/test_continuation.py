from types import SimpleNamespace

import pytest
import torch

from compute_trainium.continuation import prefix_fingerprint, validate_resume_signature, validate_resume_schedule


def schedule(start=None,end=None,peak=.001):
    return dict(peak=peak,warmup_tokens=10,decay_start=start,decay_end=end,floor_ratio=.1)


def test_only_unannealed_history_can_extend_without_resetting_optimization():
    validate_resume_schedule(schedule(90,100),schedule(270,300),90,extend_stable=True)
    validate_resume_schedule(schedule(),schedule(270,300),90,extend_stable=True)
    validate_resume_schedule(schedule(),schedule(90,100),90,decay_branch=True)
    for old,new,exposure in [(schedule(90,100),schedule(270,300),91),
                             (schedule(),schedule(270,300,peak=.002),90),
                             (schedule(),schedule(80,100),90)]:
        with pytest.raises(ValueError):validate_resume_schedule(old,new,exposure,extend_stable=True)


def test_extension_preserves_every_original_token_mask_and_exposure():
    tokens=torch.arange(17,dtype=torch.int32);domains=torch.ones(17,dtype=torch.uint8)
    old_hash=prefix_fingerprint(tokens,domains,9)
    old=dict(context=4,seed=23,corpus_sha256='old',packing_sha256=old_hash,world=64)
    new=dict(old,corpus_sha256='new',packing_sha256='new-pack')
    metadata={'prefix_parent':{'fingerprint':old_hash,'source_sha256':'old','windows':2}}
    packed=SimpleNamespace(tokens=tokens,domains=domains,prefix_counts=torch.tensor([[0],[4],[8],[12],[16]]))
    validate_resume_signature(old,new,allow_extension=True,metadata=metadata,packed=packed,cursor=2,exposure=8)
    with pytest.raises(ValueError):validate_resume_signature(old,new,allow_extension=True,metadata=metadata,packed=packed,cursor=2,exposure=9)
    with pytest.raises(ValueError):validate_resume_signature(old,dict(new,world=32),allow_extension=True,metadata=metadata,packed=packed,cursor=2,exposure=8)
    tokens[2]=999
    with pytest.raises(ValueError):validate_resume_signature(old,new,allow_extension=True,metadata=metadata,packed=packed,cursor=2,exposure=8)
