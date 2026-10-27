from scripts.prepare_workspace_source_views import anchor_positions


def test_source_only_anchor_selection_is_bounded_deterministic_and_in_spaced_bins():
    for length in [160,513,8192,16384]:
        anchors=anchor_positions(length,'fixed-document-id','train')
        assert anchors==anchor_positions(length,'fixed-document-id','train')
        assert 0<len(anchors)<=32 and len(set((p-128)//256 for p in anchors))==len(anchors)
        assert all(p>=128 and p+32<=length for p in anchors)
        dev=anchor_positions(length,'fixed-document-id','dev')
        assert len(dev)==len(set(dev)) and len(dev)<=16 and all(p>=128 and p+32<=length for p in dev)
    assert anchor_positions(159,'short','train')==[]
