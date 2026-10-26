"""Preserve immutable completed-run array bytes while sharing duplicates."""
import os
from pathlib import Path
import stat

from .training import file_sha256


def array_record(root, path):
    root, path = Path(root).resolve(), Path(path)
    info = path.lstat()
    if (not path.resolve().is_relative_to(root/'runs') or path.is_symlink()
            or not stat.S_ISREG(info.st_mode) or path.suffix != '.npy'
            or not path.name.startswith('h0') or not (path.parent/'report.json').is_file()):
        raise ValueError('Only regular h0 arrays from completed run receipts are eligible')
    return dict(path=str(path.relative_to(root)), size=info.st_size,
        sha256=file_sha256(path), device=info.st_dev, inode=info.st_ino,
        links=info.st_nlink, uid=info.st_uid, gid=info.st_gid,
        mode=stat.S_IMODE(info.st_mode), mtime_ns=info.st_mtime_ns)


def duplicate_actions(records):
    groups = {}
    for row in records:
        groups.setdefault((row['device'], row['size'], row['sha256'],
                           row['uid'], row['gid'], row['mode']), []).append(row)
    actions = []
    for rows in groups.values():
        rows = sorted(rows, key=lambda r: r['path'])
        original = rows[0]
        for row in rows[1:]:
            if row['inode'] != original['inode']:
                actions.append(dict(canonical=original, duplicate=row))
    return actions


def share_identical_array(root, action):
    """Revalidate both bytes before atomically replacing an immutable alias.

    This never targets model/checkpoint/source/corpus files. Readers retain
    the same bytes and paths. Per-path original metadata remains in the plan;
    sharing assumes the already-closed run identities stay immutable.
    """
    root = Path(root)
    canonical, duplicate = action['canonical'], action['duplicate']
    paths = [root/r['path'] for r in (canonical, duplicate)]
    for path, expected in zip(paths, (canonical, duplicate)):
        observed = array_record(root, path)
        if any(observed[k] != expected[k] for k in ('size', 'sha256', 'device', 'inode', 'uid', 'gid', 'mode')):
            raise AssertionError('Closed-array bytes or original identity changed')
    if canonical['size'] != duplicate['size'] or canonical['sha256'] != duplicate['sha256']:
        raise AssertionError('Byte-identical immutable arrays required')
    temporary = paths[1].with_name('.'+paths[1].name+'.share-identical')
    if temporary.exists(): raise FileExistsError('Preserve an interrupted atomic alias operation')
    os.link(paths[0], temporary)
    try:
        if file_sha256(temporary) != duplicate['sha256']:
            raise AssertionError('Temporary hard link changed array bytes')
        os.replace(temporary, paths[1])
    finally:
        if temporary.exists(): temporary.unlink()
    if file_sha256(paths[1]) != duplicate['sha256'] or paths[0].stat().st_ino != paths[1].stat().st_ino:
        raise AssertionError('Final identical-array alias verification failed')
    return dict(path=duplicate['path'], sha256=duplicate['sha256'], size=duplicate['size'],
                original_inode=duplicate['inode'], shared_inode=paths[1].stat().st_ino)
