from types import SimpleNamespace
import pytest
from compute_trainium.inode_reserve import require_inode_reserve

def test_inode_guard_fails_with_bytes_still_available(monkeypatch):
    monkeypatch.setattr('compute_trainium.inode_reserve.os.statvfs',lambda p:SimpleNamespace(f_favail=0,f_bavail=10**12))
    with pytest.raises(RuntimeError):require_inode_reserve(['/scratch'])
    monkeypatch.setattr('compute_trainium.inode_reserve.os.statvfs',lambda p:SimpleNamespace(f_favail=10000))
    require_inode_reserve(['/scratch'])
