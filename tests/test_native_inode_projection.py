import pytest
from compute_trainium.native_inode_projection import project_snapshot_inodes


def test_projection_preserves_guard_after_every_planned_source_copy(tmp_path):
    directory=tmp_path/'scripts';directory.mkdir();(directory/'worker.py').write_text('value=1\n')
    cache=directory/'__pycache__';cache.mkdir();(cache/'worker.cpython-312.pyc').write_bytes(b'cache-fixture')
    one=project_snapshot_inodes(tmp_path,1);four=project_snapshot_inodes(tmp_path,4)
    assert four['projected_new_inodes']>one['projected_new_inodes']
    assert four['required_free_inodes']-four['projected_new_inodes']==10000
    assert four['source_files']==2 and four['python_sources']==1


@pytest.mark.parametrize('phases',[0,17,True])
def test_reject_unbounded_or_ambiguous_phase_counts(phases,tmp_path):
    with pytest.raises(ValueError):project_snapshot_inodes(tmp_path,phases)
