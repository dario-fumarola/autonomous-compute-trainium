"""Lossless relocation of one named closed qualification array out of tmpfs."""
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def main():
    root=Path('/run/user/1000/autonomous-compute/instance-nvme')
    source=Path('/dev/shm/dario-real-cpt-scaffold-v29-qualification-v1/run-v1/step-0000008/model.pt')
    folder=source.parent.parent
    out=root/'closed-qualification-tmpfs-relocation-v1';out.mkdir(mode=0o700)
    def write(name,data):
        (out/name).write_text(json.dumps(data,indent=2)+'\n')
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        return h.hexdigest()
    start=time.time();r=dict(status='PREPARING_EXACT_QUALIFICATION_RELOCATION',assigned_chips=0,arrays=1,
        scope='One completed old qualification model moves from tmpfs to NVMe with verified bytes and an original-path symlink. No model content, scientific history, optimizer state, RNG or source data is discarded. Exact restoration is available; a future container must expose the archive path or first restore the original regular file.',
        exact_local_byte_restoration_available=True,scientific_states_deleted=0)
    write('report.json',r)
    try:
        with (folder.parent/'.run-v1.writer.lock').open('r+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            completion=json.loads((folder/'completion.json').read_text())
            if not completion or source.is_symlink() or not source.is_file() or source.stat().st_nlink!=1:
                raise RuntimeError('Named closed regular single-link file required')
            size=source.stat().st_size
            if size!=3441248275 or time.time()-source.stat().st_mtime<86400:
                raise RuntimeError('Named historical size/age differs')
            if shutil.disk_usage(root).free-size<35*1024**3:
                raise RuntimeError('Protected disk reserve')
            digest=sha(source);target=out/'model.pt'
            write('PRIVATE-plan.json',dict(source=str(source),target=str(target),sha256=digest,bytes=size,completion=completion,seconds_limit=600))
            shutil.copy2(source,target)
            with target.open('rb') as f:os.fsync(f.fileno())
            if sha(target)!=digest or sha(source)!=digest or time.time()-start>600:
                raise RuntimeError('Exact copy or time gate failed; source retained')
            temporary=source.with_name('model.pt.relocation-link')
            temporary.symlink_to(target)
            os.replace(temporary,source)
            if not source.is_symlink() or sha(source)!=digest:
                raise RuntimeError('Original-path resolution gate')
            r.update(status='COMPLETE_EXACT_QUALIFICATION_RELOCATION',bytes=size,sha256=digest,
                restored_via_original_path=True,tmpfs_bytes_released=size)
    except BaseException as e:
        r.update(status='STOPPED_QUALIFICATION_RELOCATION',error_type=type(e).__name__,error=str(e));raise
    finally:
        r['wall_seconds']=time.time()-start;write('report.json',r)


if __name__=='__main__':main()
