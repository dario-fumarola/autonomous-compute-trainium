"""Filesystem software fixtures, never model research data."""
import os
import pytest

from compute_trainium.closed_array_storage import array_record, duplicate_actions, share_identical_array


def fixture(root, name, contents=b'numerical-storage-fixture'):
    folder = root/'runs'/name; folder.mkdir(parents=True)
    (folder/'report.json').write_text('{"status":"COMPLETE"}')
    path = folder/'h0-u0-rank0.npy'; path.write_bytes(contents)
    return path


def test_identical_arrays_retain_both_paths_and_exact_bytes(tmp_path):
    a = fixture(tmp_path, 'a'); b = fixture(tmp_path, 'b')
    records = [array_record(tmp_path, p) for p in (a, b)]
    result = share_identical_array(tmp_path, duplicate_actions(records)[0])
    assert a.read_bytes() == b.read_bytes() == b'numerical-storage-fixture'
    assert a.stat().st_ino == b.stat().st_ino
    assert result['sha256'] == records[0]['sha256']


def test_changed_duplicate_is_rejected_before_mutation(tmp_path):
    a = fixture(tmp_path, 'a'); b = fixture(tmp_path, 'b')
    action = duplicate_actions([array_record(tmp_path, p) for p in (a, b)])[0]
    a_inode, b_inode = a.stat().st_ino, b.stat().st_ino
    b.write_bytes(b'changed')
    with pytest.raises(AssertionError): share_identical_array(tmp_path, action)
    assert (a.stat().st_ino, b.stat().st_ino) == (a_inode, b_inode)


def test_non_arrays_unfinished_runs_and_symlinks_are_excluded(tmp_path):
    a = fixture(tmp_path, 'a')
    checkpoint = a.with_name('checkpoint.pt'); checkpoint.write_bytes(a.read_bytes())
    with pytest.raises(ValueError): array_record(tmp_path, checkpoint)
    link = a.with_name('h0-link.npy'); os.symlink(a, link)
    with pytest.raises(ValueError): array_record(tmp_path, link)
    (a.parent/'report.json').unlink()
    with pytest.raises(ValueError): array_record(tmp_path, a)
