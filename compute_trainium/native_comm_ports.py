"""Reserve distinct job communication ports outside host ephemeral allocation."""
import fcntl
import json
import socket
from pathlib import Path


def reserve_ports(root,identities):
    root=Path(root)
    if not identities or len(set(identities))!=len(identities):raise ValueError('Unique explicit phase identities required')
    low,high=map(int,Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    registry=root/'native-comm-ports.json';result={}
    with (root/'native-comm-ports.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        saved=json.loads(registry.read_text()) if registry.exists() else {}
        if any(name in saved for name in identities):raise FileExistsError('No silent phase port reuse')
        used=set(saved.values())
        for identity in identities:
            for port in range(16000,30000):
                if port in used or low<=port<=high:continue
                probe=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                try:probe.bind(('127.0.0.1',port))
                except OSError:continue
                finally:probe.close()
                result[identity]=port;used.add(port);break
            else:raise RuntimeError('No free non-ephemeral local communication port')
        saved.update(result)
        temporary=registry.with_suffix('.tmp');temporary.write_text(json.dumps(saved,indent=2)+'\n');temporary.replace(registry)
    return result
