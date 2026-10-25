"""Keep enough directory/file capacity to write terminal receipts on failure."""
import os

def require_inode_reserve(paths,minimum=10000):
    if type(minimum) is not int or minimum<1:raise ValueError('Positive inode reserve required')
    for path in paths:
        if os.statvfs(path).f_favail<minimum:raise RuntimeError('Filesystem inode reserve failed: '+str(path))
