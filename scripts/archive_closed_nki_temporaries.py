"""Losslessly archive a bounded set of old compiler-only temporary directories."""
import argparse,hashlib,json,os,shutil,stat,subprocess,tarfile,time
from pathlib import Path


def digest_stream(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def inactive():
    value=subprocess.run(['docker','ps','-q','--filter','label=ac-trainium-managed=1'],check=True,capture_output=True,text=True)
    if value.stdout.strip():raise RuntimeError('Compiler maintenance requires no active managed worker')


def main(root,identity="probability-old-nki-archive-v1",count=4000):
    if not identity.startswith('probability-old-nki-archive-') or '/' in identity or not 1<=count<=12000:raise ValueError('Bounded immutable archive identity/count')
    start=time.time();inactive();out=root/identity
    out.mkdir(mode=0o700,exist_ok=False);tmp=root/'runtime/tmp';now=time.time()
    choices=sorted((p for p in tmp.iterdir() if p.name.startswith('nki_') and not p.is_symlink() and p.is_dir()
                    and now-p.stat().st_mtime>48*3600),key=lambda p:p.stat().st_mtime)[:count]
    if len(choices)!=count:raise RuntimeError('Exact bounded directory inventory required')
    records=[];directories=[]
    for directory in choices:
        for path in [directory,*directory.rglob('*')]:
            info=path.lstat()
            if stat.S_ISDIR(info.st_mode):directories.append(str(path.relative_to(tmp)));continue
            if not stat.S_ISREG(info.st_mode) or path.suffix!='.colz' or info.st_nlink!=1:
                raise ValueError('Only old singly-linked compiler .colz files and directories')
            with path.open('rb') as f:h=digest_stream(f)
            records.append(dict(path=str(path.relative_to(tmp)),bytes=info.st_size,sha256=h,inode=info.st_ino))
    total=sum(r['bytes'] for r in records)
    if total>1536*1024**2 or shutil.disk_usage(root).free<35*1024**3+total+64*1024**2:raise RuntimeError('Archive bound/reserve')
    plan=dict(registered_unix=start,seconds_limit=600,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files=records,directories=directories,scope='Old compiler .colz temporary files only;exact byte-restorable tar archive before removing loose copies. No weights,optimizer,source corpus,source snapshot,NEFF executable or HLO cache removed.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');before=os.statvfs(root).f_favail
    archive=out/'PRIVATE-old-nki.tar'
    with tarfile.open(archive,'w') as tar:
        for path in choices:tar.add(path,arcname=path.name,recursive=True)
    expected={r['path']:r for r in records}
    with tarfile.open(archive,'r') as tar:
        members=tar.getmembers();actual={m.name for m in members if m.isfile()}
        if actual!=set(expected) or {m.name for m in members if m.isdir()}!=set(directories):raise AssertionError('Archive inventory mismatch')
        for member in members:
            if member.isfile():
                with tar.extractfile(member) as stream:h=digest_stream(stream)
                if h!=expected[member.name]['sha256']:raise AssertionError('Archive byte mismatch')
            elif not member.isdir():raise AssertionError('Unexpected archive object')
    inactive()
    for row in records:
        path=tmp/row['path'];info=path.stat()
        with path.open('rb') as f:h=digest_stream(f)
        if info.st_ino!=row['inode'] or h!=row['sha256']:raise AssertionError('Temporary changed before removal')
    for i,path in enumerate(choices):
        if time.time()-start>600:raise TimeoutError('Bounded archive maintenance')
        if i%64==0:inactive()
        shutil.rmtree(path)
    with archive.open('rb') as f:archive_sha=digest_stream(f)
    report=dict(status='COMPLETE_EXACT_OLD_COMPILER_TEMPORARY_ARCHIVE',files=len(records),directories=len(directories),
        original_bytes=total,archive_bytes=archive.stat().st_size,archive_sha256=archive_sha,
        free_inodes_before=before,free_inodes_after=os.statvfs(root).f_favail,wall_seconds=time.time()-start,
        exact_local_byte_restoration_available=True,scientific_states_modified=0,assigned_chips=0,
        scope=plan['scope'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',default='probability-old-nki-archive-v1');p.add_argument('--count',type=int,default=4000);a=p.parse_args();main(a.root,a.identity,a.count)
