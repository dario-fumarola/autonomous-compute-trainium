"""Byte-preserving coalescing of immutable closed source snapshots."""
import json
import os
from pathlib import Path,PurePosixPath
import stat

from .training import file_sha256


def source_record(root,snapshot,relative,expected):
    root,snapshot=Path(root).resolve(),Path(snapshot)
    item=PurePosixPath(relative)
    if (snapshot.is_symlink() or snapshot.parent!=root/'snapshots'
            or item.is_absolute() or '..' in item.parts or item.suffix!='.py'
            or not (snapshot/'launch.json').is_file() or not (snapshot/'closure.json').is_file()):
        raise ValueError('Only manifest-listed Python sources from closed snapshots are eligible')
    path=snapshot/relative
    info=path.lstat()
    if path.is_symlink() or not path.resolve().is_relative_to(snapshot.resolve()) or not stat.S_ISREG(info.st_mode):
        raise ValueError('Regular in-snapshot source path required')
    if file_sha256(path)!=expected:raise AssertionError('Historical frozen source changed;preserve for diagnosis')
    return dict(path=str(path.relative_to(root)),size=info.st_size,sha256=expected,
        device=info.st_dev,inode=info.st_ino,links=info.st_nlink,uid=info.st_uid,gid=info.st_gid,
        mode=stat.S_IMODE(info.st_mode),mtime_ns=info.st_mtime_ns)


def source_actions(records):
    groups={}
    for row in records:
        groups.setdefault((row['sha256'],row['size'],row['device'],row['uid'],row['gid'],row['mode']),[]).append(row)
    actions=[]
    for rows in groups.values():
        rows.sort(key=lambda x:x['path']);canonical=rows[0]
        actions.extend(dict(canonical=canonical,duplicate=row) for row in rows[1:] if row['inode']!=canonical['inode'])
    return actions


def share_source(root,action):
    root=Path(root);left,right=action['canonical'],action['duplicate']
    if (left['sha256'],left['size'],left['device'],left['uid'],left['gid'],left['mode'])!=(right['sha256'],right['size'],right['device'],right['uid'],right['gid'],right['mode']):
        raise ValueError('Only byte/permission-identical immutable sources can share storage')
    paths=[]
    for row in (left,right):
        path=root/row['path'];relative=path.relative_to(root/'snapshots');snapshot=root/'snapshots'/relative.parts[0]
        file='/'.join(relative.parts[1:])
        manifest=json.loads((snapshot/'launch.json').read_text())['source_sha256']
        if manifest.get(file)!=row['sha256']:raise AssertionError('Historical launch source manifest changed')
        actual=source_record(root,snapshot,file,row['sha256'])
        if any(actual[k]!=row[k] for k in ('inode','device','uid','gid','mode','size')):
            raise AssertionError('Historical source identity changed before sharing')
        paths.append(path)
    temporary=paths[1].with_name('.'+paths[1].name+'.share-closed-source')
    if temporary.exists():raise FileExistsError('Preserve an interrupted source alias transaction')
    os.link(paths[0],temporary)
    try:os.replace(temporary,paths[1])
    finally:
        if temporary.exists():temporary.unlink()
    if file_sha256(paths[1])!=right['sha256']:raise AssertionError('Shared source lost its original bytes')
    return right['path']
