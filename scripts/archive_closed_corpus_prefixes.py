"""Exact split archive for two closed failed-build arrays; canonical bytes immutable."""
import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

ID = 'closed-corpus-prefix-archive-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def joined_sha(prefix, suffix):
    h = hashlib.sha256()
    for path in (prefix, suffix):
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8*1024**2), b''):
                h.update(block)
    return h.hexdigest()


def identity(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('Regular non-symlink files only')
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def stage(old, canonical, output, expected):
    before, source = identity(old), identity(canonical)
    if before[2] <= source[2] or sha(old) != expected['old_full_sha256'] or sha(canonical) != expected['accepted_sha256']:
        raise AssertionError('Pinned old/canonical bytes or sizes changed')
    output.mkdir(mode=0o700)
    prefix, suffix = output/'PRIVATE-prefix.u16', output/'PRIVATE-suffix.u16'
    # A hard link makes the archive independently retain its entire prefix,
    # even if a later maintenance operation renames the canonical path.
    os.link(canonical, prefix)
    with old.open('rb') as reader, suffix.open('xb') as writer:
        reader.seek(source[2])
        shutil.copyfileobj(reader, writer, 8*1024**2)
        writer.flush(); os.fsync(writer.fileno())
    if joined_sha(prefix, suffix) != expected['old_full_sha256']:
        raise AssertionError('Complete reconstructed byte stream differs')
    if identity(old) != before or identity(canonical) != source:
        raise AssertionError('Source changed while staging archive')
    s = old.stat()
    receipt = dict(old=str(old), canonical=str(canonical), old_identity=before,
                   canonical_identity=source, old_sha256=expected['old_full_sha256'],
                   prefix_sha256=sha(prefix), suffix_sha256=sha(suffix), old_bytes=s.st_size,
                   prefix_bytes=prefix.stat().st_size, suffix_bytes=suffix.stat().st_size,
                   old_mode=stat.S_IMODE(s.st_mode), old_uid=s.st_uid, old_gid=s.st_gid)
    (output/'PRIVATE-manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    for p in (output/'PRIVATE-manifest.json',):
        with p.open('rb') as f: os.fsync(f.fileno())
    fd = os.open(output, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return receipt


def retire(old, canonical, output, record):
    if identity(old) != record['old_identity'] or identity(canonical) != record['canonical_identity']:
        raise AssertionError('Closed source identity changed before retirement')
    if joined_sha(output/'PRIVATE-prefix.u16', output/'PRIVATE-suffix.u16') != record['old_sha256']:
        raise AssertionError('Verified complete split archive required')
    old.unlink()
    fd = os.open(old.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def restore(output):
    r = json.loads((output/'PRIVATE-manifest.json').read_text())
    old = Path(r['old'])
    if old.exists():
        raise FileExistsError('Never overwrite a current corpus array')
    if joined_sha(output/'PRIVATE-prefix.u16', output/'PRIVATE-suffix.u16') != r['old_sha256']:
        raise AssertionError('Archive integrity failure')
    pending = old.with_name(old.name+'.restoring')
    with pending.open('xb') as writer:
        for p in (output/'PRIVATE-prefix.u16', output/'PRIVATE-suffix.u16'):
            with p.open('rb') as reader: shutil.copyfileobj(reader, writer, 8*1024**2)
        writer.flush(); os.fsync(writer.fileno())
    if sha(pending) != r['old_sha256']:
        raise AssertionError('Restored file integrity failure')
    os.chmod(pending, r['old_mode'])
    if (pending.stat().st_uid, pending.stat().st_gid) != (r['old_uid'], r['old_gid']):
        os.chown(pending, r['old_uid'], r['old_gid'])
    os.utime(pending, ns=(r['old_identity'][3], r['old_identity'][3]))
    # Exclusive hard-link publication cannot replace a file created meanwhile.
    os.link(pending, old)
    pending.unlink()
    fd = os.open(old.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def inactive():
    value = subprocess.run(['docker', 'ps', '-q', '--filter', 'label=ac-trainium-managed=1'],
                           check=True, capture_output=True, text=True)
    if value.stdout.strip():
        raise RuntimeError('Perform retirement at a closed Native boundary')


def main(root, action, domain=None):
    out = root/ID
    if action == 'restore':
        if domain not in ('math', 'web'): raise ValueError('Choose one archived domain')
        manifest = json.loads((out/domain/'PRIVATE-manifest.json').read_text())
        if shutil.disk_usage(root).free < 35*1024**3+manifest['old_bytes']:
            raise RuntimeError('Restoration requires full temporary bytes plus reserve')
        inactive(); restore(out/domain)
        print(json.dumps(dict(status='RESTORED_EXACT_CLOSED_ARRAY', domain=domain)))
        return
    inactive(); start = time.time()
    failure_path = root/'natural-append-17000m-v3/failure.json'
    if json.loads(failure_path.read_text()).get('status') != 'STOPPED_CHAINED_DATA_PREPARATION':
        raise AssertionError('Named historical producer must remain stopped')
    audit_path = root/'old-corpus-prefix-identity-audit-v1/report.json'
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'COMPLETE_READ_ONLY_CORPUS_PREFIX_IDENTITY_AUDIT' or not all(r['prefix_equal'] for r in audit['records'].values()):
        raise AssertionError('Both complete read-only prefix audits required')
    need = sum(r['old_bytes']-r['accepted_bytes'] for r in audit['records'].values())
    if shutil.disk_usage(root).free < 35*1024**3+need+256*1024**2:
        raise RuntimeError('Suffixes plus protected disk reserve required')
    out.mkdir(mode=0o700)
    plan = dict(registered_unix=start, seconds_limit=600, audit_sha256=sha(audit_path),
                stopped_producer_receipt_sha256=sha(failure_path),
                source_sha256=sha(Path(__file__)), domains=['math', 'web'],
                scope='Exact split archival of named closed failed-build arrays. Retain accepted prefix by hard link and all unique suffix bytes; verify reconstructed complete SHA256 before unlinking only the failed-build loose arrays. No scientific state, accepted data, source index or failure receipt changed.')
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    records = {}; before = shutil.disk_usage(root).free
    for name in plan['domains']:
        old = root/'natural-append-17000m-v3/extension'/f'{name}.u16'
        canonical = root/'natural-append-capacity-v4/extension'/f'{name}.u16'
        records[name] = stage(old, canonical, out/name, audit['records'][name])
    for name, record in records.items():
        inactive()
        if time.time()-start > 600: raise TimeoutError('Archive time bound')
        retire(Path(record['old']), Path(record['canonical']), out/name, record)
    report = dict(status='COMPLETE_EXACT_CLOSED_CORPUS_SPLIT_ARCHIVE',
                  domains={n:{k:r[k] for k in ('old_bytes', 'prefix_bytes', 'suffix_bytes', 'old_sha256', 'prefix_sha256', 'suffix_sha256')} for n,r in records.items()},
                  free_bytes_before=before, free_bytes_after=shutil.disk_usage(root).free,
                  unique_bytes_discarded=0, scientific_states_modified=0,
                  assigned_chips=0, wall_seconds=time.time()-start, exact_restoration_available=True,
                  scope=plan['scope'])
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--action', choices=['archive', 'restore'], required=True)
    parser.add_argument('--domain', choices=['math', 'web'])
    args = parser.parse_args()
    main(args.root, args.action, args.domain)
