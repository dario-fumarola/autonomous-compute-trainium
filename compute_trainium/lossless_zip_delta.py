"""Exact archives sharing unchanged ZIP payloads, with full binary restoration.

Private tensor data is never decoded or published. Different ZIP headers and
pickle metadata are retained literally. Every reconstructed byte is checked
against the original whole-file SHA256 before a caller may release its copy.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import zipfile


def digest_file(path):
    with Path(path).open('rb') as stream:
        value=hashlib.sha256()
        for block in iter(lambda:stream.read(4*1024**2),b''):value.update(block)
    return value.hexdigest()


def payload_offset(stream,info):
    stream.seek(info.header_offset);header=stream.read(30)
    if len(header)!=30 or header[:4]!=b'PK\x03\x04':raise ValueError('Expected ordinary ZIP local header')
    name,extra=struct.unpack_from('<HH',header,26)
    return info.header_offset+30+name+extra


def region_hash(stream,offset,length):
    stream.seek(offset);value=hashlib.sha256()
    while length:
        block=stream.read(min(length,4*1024**2))
        if not block:raise EOFError('Truncated ZIP payload')
        value.update(block);length-=len(block)
    return value.hexdigest()


def archive_stream(recipe_path):
    recipe_path=Path(recipe_path);recipe=json.loads(recipe_path.read_text())
    if recipe['schema']!='exact_zip_payload_archive_v1':raise ValueError('Unknown archive schema')
    parent=recipe_path.parent/recipe['parent_relative'];literal=recipe_path.parent/recipe['literal_file']
    if digest_file(parent)!=recipe['parent_sha256'] or digest_file(literal)!=recipe['literal_sha256']:
        raise AssertionError('Exact archive dependency changed')
    with parent.open('rb') as shared,literal.open('rb') as own:
        for source,offset,length in recipe['segments']:
            if source not in ('parent','literal') or offset<0 or length<0:raise ValueError('Invalid archive segment')
            stream=shared if source=='parent' else own;stream.seek(offset)
            while length:
                block=stream.read(min(length,4*1024**2))
                if not block:raise EOFError('Archive segment truncated')
                yield block;length-=len(block)


def reconstructed_digest(recipe_path):
    digest=hashlib.sha256();length=0
    for block in archive_stream(recipe_path):digest.update(block);length+=len(block)
    return digest.hexdigest(),length


def create_archive(parent,child,recipe_path):
    parent,child,recipe_path=map(Path,(parent,child,recipe_path))
    literal_path=recipe_path.with_suffix(recipe_path.suffix+'.literal')
    if recipe_path.exists() or literal_path.exists():raise FileExistsError('Preserve every archive attempt')
    if parent.is_symlink() or child.is_symlink() or not parent.is_file() or not child.is_file():
        raise ValueError('Original regular files required')
    parent_hash,child_hash=digest_file(parent),digest_file(child);child_size=child.stat().st_size
    segments=[];referenced=0
    with zipfile.ZipFile(parent) as pzip,zipfile.ZipFile(child) as czip,parent.open('rb') as shared,child.open('rb') as original,literal_path.open('xb') as literal:
        parent_infos={'/'.join(info.filename.split('/')[1:]):info for info in pzip.infolist()}
        cursor=0
        def copy_literal(start,length):
            if not length:return
            offset=literal.tell();original.seek(start);remaining=length
            while remaining:
                block=original.read(min(remaining,4*1024**2))
                if not block:raise EOFError('Original file truncated')
                literal.write(block);remaining-=len(block)
            segments.append(['literal',offset,length])
        for info in sorted(czip.infolist(),key=lambda x:x.header_offset):
            start=payload_offset(original,info);length=info.compress_size
            copy_literal(cursor,start-cursor)
            key='/'.join(info.filename.split('/')[1:]);peer=parent_infos.get(key)
            matched=False
            if peer and peer.compress_type==info.compress_type and peer.compress_size==length and peer.CRC==info.CRC:
                pstart=payload_offset(shared,peer)
                matched=region_hash(shared,pstart,length)==region_hash(original,start,length)
            if matched and length:
                segments.append(['parent',pstart,length]);referenced+=length
            else:copy_literal(start,length)
            cursor=start+length
        copy_literal(cursor,child_size-cursor)
    recipe=dict(schema='exact_zip_payload_archive_v1',child_name=child.name,original_sha256=child_hash,
        original_bytes=child_size,parent_relative=os.path.relpath(parent,recipe_path.parent),parent_sha256=parent_hash,
        literal_file=literal_path.name,literal_sha256=digest_file(literal_path),segments=segments,shared_payload_bytes=referenced)
    recipe_path.write_text(json.dumps(recipe,indent=2)+'\n')
    restored,length=reconstructed_digest(recipe_path)
    if restored!=child_hash or length!=child_size:raise AssertionError('Archive failed exact original-byte reconstruction')
    if digest_file(parent)!=parent_hash or digest_file(child)!=child_hash:raise AssertionError('Original file changed during archive')
    return dict(original_sha256=child_hash,original_bytes=child_size,shared_payload_bytes=referenced,
        archive_bytes=recipe_path.stat().st_size+literal_path.stat().st_size,verified_exact_restoration=True)


def restore_archive(recipe_path,target):
    recipe_path,target=Path(recipe_path),Path(target)
    if target.exists():raise FileExistsError('Do not overwrite an existing evidence file')
    recipe=json.loads(recipe_path.read_text());value=hashlib.sha256();length=0
    with target.open('xb') as output:
        for block in archive_stream(recipe_path):output.write(block);value.update(block);length+=len(block)
    if value.hexdigest()!=recipe['original_sha256'] or length!=recipe['original_bytes']:
        raise AssertionError('Materialized archive does not match original fingerprint')
