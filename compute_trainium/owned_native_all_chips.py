"""One exclusive all-chip worker, sharing the existing half-ownership lock."""
import fcntl
import json
from pathlib import Path
import subprocess

from .execution_isolation import metadata, managed_running, verify_owner


def create_all_chip_worker(root, identity, *, memory_gib=128):
    if type(memory_gib) is not int or memory_gib not in (128,192,256):
        raise ValueError('Explicit bounded 128/192/256GiB worker required')
    if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in identity):
        raise ValueError('Explicit safe all-chip identity required')
    root = Path(root)
    source = json.loads((root/'isolated-native-env-v1/parent-private.json').read_text())
    image = json.loads((root/'isolated-native-env-v1/setup-private.json').read_text())['image']
    name = 'ac-worker-' + identity
    with (root/'native-half-reservations.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if managed_running():
            raise RuntimeError('All-chip worker requires every prior managed owner to be terminal')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<(512+memory_gib)*1024**3:
            raise RuntimeError('Protected host reserve plus complete worker allowance required')
        if subprocess.run(['docker','inspect',name],capture_output=True).returncode == 0:
            raise FileExistsError('No silent all-chip container reuse')
        # No half label: the existing half factory rejects this unclassified
        # exclusive owner. It uses the same lock, so neither direction can race.
        args = ['docker','run','-d','--name',name,'--label','ac-trainium-managed=1',
            '--label','ac-trainium-owner='+identity,'--label','ac-trainium-chips=all16',
            '--network','host','--ipc','private','--shm-size','64m',
            '--memory',str(memory_gib)+'g','--memory-swap',str(memory_gib)+'g']
        for device in source['HostConfig']['Devices']:
            args += ['--device',device['PathOnHost']+':'+device['PathInContainer']+':'+device['CgroupPermissions']]
        for mount in source['Mounts']:
            if mount['Type'] != 'bind': raise ValueError('Unexpected source mount')
            spec = 'type=bind,source='+mount['Source']+',target='+mount['Destination']
            if not mount['RW']: spec += ',readonly'
            args += ['--mount',spec]
        subprocess.run(args+[image,'sleep','infinity'],check=True,capture_output=True)
        verify_owner(metadata(name),identity)
        return name
