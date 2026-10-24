import json
import socket

import pytest

from compute_trainium import native_comm_ports
from compute_trainium.native_comm_ports import reserve_ports


def test_distinct_persistent_ports_avoid_kernel_ephemeral_range(tmp_path,monkeypatch):
    # Do not briefly bind ports reserved for concurrently running native jobs.
    class Probe:
        def bind(self,_):pass
        def close(self):pass
    monkeypatch.setattr(native_comm_ports.socket,'socket',lambda *_:Probe())
    ports=reserve_ports(tmp_path,['first','second'])
    assert len(set(ports.values()))==2
    low,high=map(int,open('/proc/sys/net/ipv4/ip_local_port_range').read().split())
    assert all(not low<=p<=high for p in ports.values())
    more=reserve_ports(tmp_path,['third'])
    assert set(ports.values()).isdisjoint(more.values())
    assert json.loads((tmp_path/'native-comm-ports.json').read_text())=={**ports,**more}
    with pytest.raises(FileExistsError):reserve_ports(tmp_path,['first'])
