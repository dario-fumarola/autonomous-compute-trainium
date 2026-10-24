"""Ownership-checked cancellation for separate native worker containers."""
import json
import subprocess
from pathlib import Path

MANAGED = 'ac-trainium-managed'
OWNER = 'ac-trainium-owner'
GLOBAL_EMERGENCIES = frozenset({'host_memory_reserve', 'filesystem_reserve'})


def metadata(container):
    return json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]


def verify_owner(data, owner):
    labels = data['Config'].get('Labels') or {}
    if labels.get(MANAGED) != '1' or labels.get(OWNER) != owner:
        raise ValueError('Refusing cancellation/launch outside the declared owner')


def managed_running():
    return subprocess.check_output(['docker', 'ps', '-q', '--filter', 'label='+MANAGED+'=1'], text=True).split()


def remaining_memory_allowance():
    total = 0
    for container in managed_running():
        data = metadata(container)
        limit = data['HostConfig']['Memory']
        if limit <= 0: raise ValueError('Every managed container needs a finite memory cap')
        pid = data['State']['Pid']
        cg = Path(f'/proc/{pid}/cgroup').read_text().strip().split('::', 1)[1]
        used = int((Path('/sys/fs/cgroup')/cg.lstrip('/')/'memory.current').read_text())
        total += max(0, limit-used)
    return total


def cancellation_targets(container, owner, reason):
    verify_owner(metadata(container), owner)
    # Routine compiler, timeout, numeric and per-container OOM failures remain
    # within their own container. Only shared host reserves justify global stop.
    return managed_running() if reason in GLOBAL_EMERGENCIES else [container]


def cancel_worker_container(container, owner, reason):
    targets = cancellation_targets(container, owner, reason)
    if targets:
        subprocess.run(['docker', 'stop', '--time', '5', *targets], capture_output=True,
                       check=True, timeout=30)
    return len(targets)
