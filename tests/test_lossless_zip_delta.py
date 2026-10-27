import zipfile

import pytest

from compute_trainium.lossless_zip_delta import create_archive,restore_archive,digest_file


def test_different_metadata_and_archive_headers_restore_every_original_byte(tmp_path):
    parent=tmp_path/'parent.pt';child=tmp_path/'child.pt'
    payload=bytes(range(256))*4096
    with zipfile.ZipFile(parent,'w') as archive:
        archive.writestr('first-root/data/0',payload);archive.writestr('first-root/data.pkl',b'old metadata')
    with zipfile.ZipFile(child,'w') as archive:
        archive.writestr('different-root/data/0',payload);archive.writestr('different-root/data.pkl',b'new metadata and order')
    original=child.read_bytes();recipe=tmp_path/'child.archive.json'
    row=create_archive(parent,child,recipe)
    assert row['verified_exact_restoration'] and row['archive_bytes']<len(original)//100
    child.unlink();restore_archive(recipe,child)
    assert child.read_bytes()==original
    with pytest.raises(FileExistsError):restore_archive(recipe,child)
    parent.write_bytes(b'changed source dependency')
    with pytest.raises(AssertionError,match='dependency changed'):restore_archive(recipe,tmp_path/'must-not-restore.pt')
