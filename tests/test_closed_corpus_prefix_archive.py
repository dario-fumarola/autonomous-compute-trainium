import pytest

from scripts.archive_closed_corpus_prefixes import stage, retire, restore, sha


def test_archive_retains_complete_bytes_without_canonical_path_dependency(tmp_path):
    canonical = tmp_path/'canonical.u16'; old = tmp_path/'closed.u16'
    canonical.write_bytes(bytes(range(256))*3)
    original = canonical.read_bytes()+b'unique suffix'
    old.write_bytes(original)
    out = tmp_path/'archive'
    expected = dict(old_full_sha256=sha(old), accepted_sha256=sha(canonical))
    row = stage(old, canonical, out, expected)
    assert old.read_bytes() == original and canonical.read_bytes() == original[:-13]
    assert canonical.stat().st_ino == (out/'PRIVATE-prefix.u16').stat().st_ino
    retire(old, canonical, out, row)
    assert not old.exists()
    canonical.unlink()  # Archive's hard link independently retains all bytes.
    restore(out)
    assert old.read_bytes() == original
    with pytest.raises(FileExistsError): restore(out)


def test_wrong_prefix_never_retires_source(tmp_path):
    old, canonical = tmp_path/'old', tmp_path/'canonical'
    old.write_bytes(b'abcdef'); canonical.write_bytes(b'wrong')
    with pytest.raises(AssertionError):
        stage(old, canonical, tmp_path/'archive', dict(old_full_sha256=sha(old), accepted_sha256=sha(canonical)))
    assert old.read_bytes() == b'abcdef'


def test_changed_archive_or_source_fails_closed(tmp_path):
    old, canonical = tmp_path/'old', tmp_path/'canonical'
    canonical.write_bytes(b'abc'); old.write_bytes(b'abcdef')
    out = tmp_path/'archive'
    row = stage(old, canonical, out, dict(old_full_sha256=sha(old), accepted_sha256=sha(canonical)))
    (out/'PRIVATE-suffix.u16').write_bytes(b'corrupt')
    with pytest.raises(AssertionError): retire(old, canonical, out, row)
    assert old.read_bytes() == b'abcdef'
