from scripts.sparsify_closed_numerical_artifacts import recorded_roles


def test_only_named_artifacts_with_recorded_hashes_can_be_selected():
    sha='a'*64
    assert recorded_roles({'gradient_sha256':sha,'unregistered.pt':sha})=={'gradient.pt':sha}
    assert not recorded_roles({'gradient_sha256':None,'checkpoint_sha256':'short'})
