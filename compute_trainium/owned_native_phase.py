"""Ephemeral per-phase containers with atomic half-node/memory reservations."""
import fcntl
import json
import subprocess
from pathlib import Path

from .execution_isolation import metadata, managed_running, verify_owner


def create_worker(root, identity, half):
    if half not in [0, 1] or not identity or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in identity):
        raise ValueError('Explicit safe phase identity and half required')
    root = Path(root)
    source = json.loads((root/'isolated-native-env-v1/parent-private.json').read_text())
    image = json.loads((root/'isolated-native-env-v1/setup-private.json').read_text())['image']
    name = 'ac-worker-'+identity
    with (root/'native-half-reservations.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active = [metadata(c) for c in managed_running()]
        if any((c['Config'].get('Labels') or {}).get('ac-trainium-half') in [None, str(half)] for c in active):
            raise RuntimeError('Requested half has an existing or unclassified managed owner')
        if sum(c['HostConfig']['Memory'] for c in active)+160*1024**3 > 320*1024**3:
            raise RuntimeError('Aggregate native320GiB cap would be exceeded')
        if subprocess.run(['docker', 'inspect', name], capture_output=True).returncode == 0:
            raise FileExistsError('No silent worker-container reuse')
        args = ['docker', 'run', '-d', '--name', name, '--label', 'ac-trainium-managed=1',
                '--label', 'ac-trainium-owner='+identity, '--label', 'ac-trainium-half='+str(half),
                '--network', 'host', '--ipc', 'private', '--shm-size', '64m',
                '--memory', '160g', '--memory-swap', '160g']
        for device in source['HostConfig']['Devices']:
            args += ['--device', device['PathOnHost']+':'+device['PathInContainer']+':'+device['CgroupPermissions']]
        for mount in source['Mounts']:
            if mount['Type'] != 'bind': raise ValueError('Unexpected source mount')
            spec = 'type=bind,source='+mount['Source']+',target='+mount['Destination']
            if not mount['RW']: spec += ',readonly'
            args += ['--mount', spec]
        subprocess.run(args+[image, 'sleep', 'infinity'], check=True, capture_output=True)
        verify_owner(metadata(name), identity)
        return name
