"""One exact per-rank checkpoint without staging a full CPU model/optimizer.

Each tensor is serialized independently into one ZIP file;only one local
matrix is copied to CPU at a time. Metadata/coverage and content hashes are
verified before atomic publication and again on restore. No lossy conversion.
"""
import hashlib
import io
import os
from pathlib import Path
import torch
import zipfile


def tensor_hash(value):
    data=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def save_streaming(path,tensors,metadata,*,compress=False):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.incomplete')
    if path.exists() or temporary.exists():raise FileExistsError('Never overwrite a preserved full state')
    keys=[];signatures={}
    try:
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED,
                             compresslevel=1 if compress else None,allowZip64=True) as archive:
            for index,(name,value) in enumerate(tensors):
                if name in signatures:raise ValueError('Duplicate checkpoint tensor')
                cpu=value.detach().cpu().contiguous();member=f'tensor-{index}.pt'
                signatures[name]=dict(member=member,sha256=tensor_hash(cpu),shape=tuple(cpu.shape),dtype=str(cpu.dtype))
                keys.append(name)
                with archive.open(member,'w',force_zip64=True) as output:torch.save(cpu,output)
                del cpu
            with archive.open('metadata.pt','w',force_zip64=True) as output:
                torch.save(dict(order=keys,signatures=signatures,metadata=metadata),output)
        with zipfile.ZipFile(temporary) as archive:
            receipt=torch.load(io.BytesIO(archive.read('metadata.pt')),weights_only=True)
            for name in receipt['order']:
                sig=receipt['signatures'][name]
                value=torch.load(io.BytesIO(archive.read(sig['member'])),weights_only=True)
                if tensor_hash(value)!=sig['sha256']:raise AssertionError('Full-state write validation failed')
                del value
        os.replace(temporary,path)
        return dict(tensors=len(keys),bytes=path.stat().st_size,metadata=metadata)
    except BaseException:
        # Preserve an incomplete archive for diagnosis;never silently retry it.
        raise


def restore_streaming(path,tensors):
    with zipfile.ZipFile(path) as archive:
        receipt=torch.load(io.BytesIO(archive.read('metadata.pt')),weights_only=True)
        count=0
        with torch.no_grad():
            for name,target in tensors:
                if count>=len(receipt['order']) or receipt['order'][count]!=name:raise ValueError('Exact model/optimizer coverage and order required')
                sig=receipt['signatures'][name]
                value=torch.load(io.BytesIO(archive.read(sig['member'])),weights_only=True)
                if tuple(value.shape)!=tuple(sig['shape']) or str(value.dtype)!=sig['dtype'] or tensor_hash(value)!=sig['sha256']:raise AssertionError('Corrupt checkpoint tensor')
                if target.shape!=value.shape or target.dtype!=value.dtype:raise ValueError('No dtype or shape conversion on restore')
                target.copy_(value.to(target.device))
                if tensor_hash(target)!=sig['sha256']:raise AssertionError('Restored tensor differs')
                count+=1;del value
        if count!=len(receipt['order']):raise ValueError('Missing checkpoint tensors')
    return receipt['metadata']
