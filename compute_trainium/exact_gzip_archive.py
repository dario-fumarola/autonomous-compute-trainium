"""Lossless binary archives with verified original-byte restoration."""
import gzip
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):value.update(block)
    return value.hexdigest()


def verified_stream(recipe):
    recipe=Path(recipe);record=json.loads(recipe.read_text());file=recipe.parent/record['archive_file']
    if record['schema']!='exact_gzip_binary_v1' or sha256_file(file)!=record['archive_sha256']:
        raise AssertionError('Exact compressed archive changed')
    value=hashlib.sha256();length=0
    with gzip.open(file,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            value.update(block);length+=len(block);yield block
    if value.hexdigest()!=record['original_sha256'] or length!=record['original_bytes']:
        raise AssertionError('Restored binary fingerprint differs')


def create_archive(source):
    source=Path(source);archive=source.with_suffix(source.suffix+'.exact.gz');recipe=source.with_suffix(source.suffix+'.exact.json')
    if source.is_symlink() or not source.is_file() or archive.exists() or recipe.exists():raise ValueError('New exact archive of an original regular file required')
    original=sha256_file(source);size=source.stat().st_size
    with source.open('rb') as incoming,archive.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as compressed:
            for block in iter(lambda:incoming.read(4*1024**2),b''):compressed.write(block)
    record=dict(schema='exact_gzip_binary_v1',original_file=source.name,original_sha256=original,original_bytes=size,
        archive_file=archive.name,archive_sha256=sha256_file(archive),archive_bytes=archive.stat().st_size)
    recipe.write_text(json.dumps(record,indent=2)+'\n')
    for _ in verified_stream(recipe):pass
    if sha256_file(source)!=original:raise AssertionError('Original binary changed during archival')
    return recipe,record


def restore_archive(recipe,target):
    target=Path(target)
    if target.exists() or target.is_symlink():raise FileExistsError('Never overwrite existing evidence')
    temporary=target.with_name(target.name+'.restore-partial')
    with temporary.open('xb') as stream:
        for block in verified_stream(recipe):stream.write(block)
    os.link(temporary,target)  # Atomic exclusive creation; a racing file wins.
    temporary.unlink()
