"""Preserve two exact old qualification shards on disk to free host RAM."""
import fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    root=Path('/run/user/1000/autonomous-compute/instance-nvme')
    folder=Path('/dev/shm/dario-real-cpt-scaffold-v29-qualification-v1/run-v1')
    stage=folder/'step-0000008';owner=folder.stat();out=root/'closed-qualification-tmpfs-relocation-v2'
    out.mkdir(mode=0o700);os.chown(out,owner.st_uid,owner.st_gid);start=time.time()
    r=dict(status='PREPARING_EXACT_CLOSED_SHARD_RELOCATION',files=[],assigned_chips=0,
        scientific_states_deleted=0,exact_local_byte_restoration_available=True,
        scope='Two named completed old qualification shards move from tmpfs to NVMe with original recorded hashes and original-path symlinks. All tensor/RNG/optimizer bytes remain. Original-owner access is preserved; a future container must expose the archive path or restore the original regular files.')
    def write():
        p=out/'report.json';p.write_text(json.dumps(r,indent=2)+'\n');os.chown(p,owner.st_uid,owner.st_gid)
    write()
    try:
        with (folder.parent/'.run-v1.writer.lock').open('r+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            completion=json.loads((folder/'completion.json').read_text())
            if completion.get('status')!='complete' or completion.get('purpose')!='real_cpt_scaffold_qualification' or completion.get('actual_steps')!=completion.get('planned_steps'):
                raise AssertionError('Original completed qualification required')
            manifest=json.loads((stage/'COMPLETE.json').read_text())
            # The preceding model relocation must remain readable by its original
            # owner, not merely by root. Repair directory/file ownership only.
            prior=stage/'model.pt';expected=root/'closed-qualification-tmpfs-relocation-v1/model.pt'
            if not prior.is_symlink() or prior.resolve()!=expected or sha(expected)!=manifest['files']['model.pt']['sha256']:
                raise AssertionError('Exact previous model relocation required')
            os.chown(expected.parent,owner.st_uid,owner.st_gid);os.chown(expected,owner.st_uid,owner.st_gid)
            r['prior_model_original_owner_access_restored']=True;write()
            for name in ('rank-00000.pt','rank-00001.pt'):
                source=stage/name;record=manifest['files'][name];info=source.stat()
                if source.is_symlink() or not source.is_file() or info.st_nlink!=1 or info.st_size!=record['bytes'] or time.time()-info.st_mtime<86400:
                    raise AssertionError('Named old single-link regular shard required')
                used=subprocess.run(['fuser','-s',str(source)],check=False).returncode
                if used!=1:raise RuntimeError('Open shard or unexpected fuser error')
                if sha(source)!=record['sha256']:raise AssertionError('Original byte receipt differs')
                if shutil.disk_usage(root).free-record['bytes']<35*1024**3:raise RuntimeError('Protected disk reserve')
                target=out/name;shutil.copy2(source,target);os.chown(target,info.st_uid,info.st_gid)
                with target.open('rb') as f:os.fsync(f.fileno())
                if sha(target)!=record['sha256'] or sha(source)!=record['sha256'] or time.time()-start>600:
                    raise AssertionError('Exact copy/time guard failed; original retained')
                link=source.with_name(name+'.relocation-link');link.symlink_to(target);os.replace(link,source)
                for directory in (out,stage):
                    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
                if sha(source)!=record['sha256']:raise AssertionError('Original-path byte verification')
                r['files'].append(dict(name=name,**record,original_owner_preserved=True));write()
            r.update(status='COMPLETE_EXACT_CLOSED_SHARD_RELOCATION',tmpfs_bytes_released=sum(x['bytes'] for x in r['files']))
    except BaseException as e:
        r.update(status='STOPPED_CLOSED_SHARD_RELOCATION',error_type=type(e).__name__,error=str(e));raise
    finally:r['wall_seconds']=time.time()-start;write()


if __name__=='__main__':main()
