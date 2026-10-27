import json
from pathlib import Path
import numpy as np
import pytest
from compute_trainium.training import file_sha256
from scripts.freeze_scale_body_exposure import first_full_update_at_least
from scripts.run_scale_main import boundary_scope_files


def test_full_global_update_rounding_never_uses_partial_or_unaccepted_data():
    counts=np.array([[i*3,i*2,i] for i in range(13)],dtype=np.int64)
    assert first_full_update_at_least(counts,25,global_windows=4)==2
    assert first_full_update_at_least(counts,24,global_windows=4)==1
    with pytest.raises(ValueError,match='capacity'):first_full_update_at_least(counts,73,global_windows=4)


def test_superseded_unrun_main_cannot_silently_use_old_endpoint(tmp_path):
    original=tmp_path/'scale-boundary-scope-freeze-v1/scope.json'
    original.parent.mkdir();original.write_text(json.dumps({'scope_sha256':file_sha256('docs/SCALE_BOUNDARY_RATE_COMPLETION_20261027.md')}))
    assert boundary_scope_files(tmp_path)[1]==original
    revised=tmp_path/'scale-body-exposure-scope-v1/scope.json';revised.parent.mkdir()
    revised.write_text(json.dumps(dict(original_unadmitted_boundary_scope_sha256=file_sha256(original),
        scope_sha256=file_sha256('docs/SCALE_BODY_EXPOSURE_DEVELOPMENT_20261027.md'))))
    with pytest.raises(ValueError,match='superseded'):boundary_scope_files(tmp_path)
    assert boundary_scope_files(tmp_path,body_exposure=True)[1]==revised
    revised.write_text(json.dumps(dict(original_unadmitted_boundary_scope_sha256='changed',
        scope_sha256=file_sha256('docs/SCALE_BODY_EXPOSURE_DEVELOPMENT_20261027.md'))))
    with pytest.raises(AssertionError,match='Original precision'):boundary_scope_files(tmp_path,body_exposure=True)
