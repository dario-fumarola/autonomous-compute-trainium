import json

import pytest

from compute_trainium.exact_gzip_archive import create_archive,restore_archive,verified_stream


def test_exact_binary_restoration_and_corruption_rejection(tmp_path):
    original=bytes(range(256))*1024+b'non-aligned tail'
    source=tmp_path/'state.pt';source.write_bytes(original)
    recipe,record=create_archive(source);target=tmp_path/'restored.pt';restore_archive(recipe,target)
    assert target.read_bytes()==original and record['original_bytes']==len(original)
    with pytest.raises(FileExistsError):restore_archive(recipe,target)
    archive=tmp_path/record['archive_file'];content=archive.read_bytes();archive.write_bytes(content[:-1]+bytes([content[-1]^1]))
    with pytest.raises(AssertionError):list(verified_stream(recipe))
