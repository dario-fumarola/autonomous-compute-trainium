"""Reconstruct ordinary CPU modules from exact streamed local-shard states."""
import io
import zipfile
import torch
from .workspace_streaming_checkpoint import tensor_hash


@torch.no_grad()
def load_streamed_model(paths,model,*,label):
    archives=[]
    try:
        archives=[zipfile.ZipFile(p) for p in paths]
        receipts=[torch.load(io.BytesIO(a.read('metadata.pt')),weights_only=True) for a in archives]
        names=[n for n,p in model.named_parameters()]
        if any(r['metadata']['parameter_names']!=names for r in receipts):raise ValueError('Exact independent model names/order required')
        for name,p in model.named_parameters():
            if p.device.type!='cpu':raise ValueError('Independent CPU reconstruction only')
            replicated=None
            for rank,(archive,receipt) in enumerate(zip(archives,receipts,strict=True)):
                sig=receipt['signatures'][label+'.'+name]
                value=torch.load(io.BytesIO(archive.read(sig['member'])),weights_only=True)
                if tensor_hash(value)!=sig['sha256']:raise AssertionError('Preserved model shard changed')
                if value.shape==p.shape:
                    if replicated is None:p.copy_(value);replicated=True
                    elif not torch.equal(p,value):raise AssertionError('Replicated parameters disagree')
                else:
                    if p.shape[0]%len(paths):raise ValueError('Only exact axis-zero sharding supported')
                    target=p.chunk(len(paths),0)[rank]
                    if target.shape!=value.shape or target.dtype!=value.dtype:raise ValueError('Model shard shape/dtype mismatch')
                    target.copy_(value)
            del value
        return dict(rank_count=len(paths),parameter_count=sum(p.numel() for p in model.parameters()))
    finally:
        for a in archives:a.close()
