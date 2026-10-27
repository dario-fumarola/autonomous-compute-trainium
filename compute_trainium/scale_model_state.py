"""Topology-bound private numerical shards, explicitly ineligible for science."""
import json
from pathlib import Path

import torch
import torch.distributed as dist

from .scale_sharded_candidate import local_parameter
from .scale_sharded_fixture import cpu_tree
from .training import atomic_checkpoint,file_sha256
from scripts.qualify_native import digest

PURPOSE='scale_large_model_qualification_only'


def validate_shard(saved,signature,rank,world,update):
    if (saved['purpose']!=PURPOSE or saved['qualification_only'] is not True
            or saved['eligible_for_science'] is not False or saved['signature']!=signature
            or saved['rank']!=rank or saved['world']!=world or saved['update']!=update
            or saved['cursor']!=update*512):
        raise ValueError('Exact same-topology qualification shard required')


def replicated_state(optimizer):
    result={}
    for name,p in optimizer.named:
        if p.placements[0].is_replicate():result[name]=local_parameter(p)
    for p,alias in zip(optimizer.groups[0]['params'],optimizer.local_adam,strict=True):
        if p.placements[0].is_replicate():
            result[optimizer.names[id(p)]+':adamw']=optimizer.adamw.state[alias]
    return result


def save_shard(model,optimizer,folder,signature,update,packed,neural_ops,optimizer_ops):
    rank,world=dist.get_rank(),dist.get_world_size();folder=Path(folder);folder.mkdir(exist_ok=True)
    value=dict(purpose=PURPOSE,qualification_only=True,eligible_for_science=False,signature=signature,
        rank=rank,world=world,update=update,cursor=update*512,real_targets=sum(packed.counts(0,update*512)),
        neural_ops=neural_ops,optimizer_ops=optimizer_ops,model_shards={n:local_parameter(p) for n,p in model.named_parameters()},
        optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state())
    cpu=cpu_tree(value);state_digest=digest(cpu)
    file=folder/f'rank{rank}-u{update}.pt';atomic_checkpoint(file,cpu)
    readback=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
    if digest(readback)!=state_digest:raise AssertionError('Private qualification shard disk readback differs')
    note=dict(rank=rank,file=file.name,sha256=file_sha256(file),state_digest=state_digest,
        update=update,cursor=update*512,real_targets=value['real_targets'],
        replicated_state_digest=digest(replicated_state(optimizer)),schema=optimizer.schema)
    (folder/f'rank{rank}-u{update}.json').write_text(json.dumps(note,indent=2)+'\n');dist.barrier()
    if rank==0:
        notes=[json.loads((folder/f'rank{i}-u{update}.json').read_text()) for i in range(world)]
        if len({r['replicated_state_digest'] for r in notes})!=1:
            raise AssertionError('Replicated parameter/moment discrepancy; all private shards retained as rejected evidence')
        if any(r['schema']!=optimizer.schema or r['update']!=update or r['cursor']!=update*512 for r in notes):
            raise AssertionError('Common sharded checkpoint metadata differs')
        (folder/f'manifest-u{update}.json').write_text(json.dumps(dict(purpose=PURPOSE,
            qualification_only=True,eligible_for_science=False,signature=signature,world=world,update=update,
            all_replicated_states_equal=True,shards=notes),indent=2)+'\n')
    dist.barrier()
    return state_digest


def load_shard(model,optimizer,folder,signature,update,packed):
    rank,world=dist.get_rank(),dist.get_world_size();folder=Path(folder)
    manifest=json.loads((folder/f'manifest-u{update}.json').read_text())
    if (manifest['signature']!=signature or manifest['world']!=world or manifest['update']!=update
            or not manifest['all_replicated_states_equal'] or manifest['eligible_for_science'] is not False):
        raise ValueError('Complete exact same-topology qualification manifest required')
    note=manifest['shards'][rank];file=folder/note['file']
    if note['rank']!=rank or file_sha256(file)!=note['sha256']:raise AssertionError('Exact rank shard changed')
    value=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
    validate_shard(value,signature,rank,world,update)
    if digest(value)!=note['state_digest'] or value['real_targets']!=sum(packed.counts(0,update*512)):
        raise AssertionError('Private qualification state/counters differ')
    device=local_parameter(next(model.parameters())).device
    with torch.no_grad():
        for name,p in model.named_parameters():local_parameter(p).copy_(value['model_shards'][name].to(device))
    optimizer.load_state_dict(value['optimizer']);torch.set_rng_state(value['cpu_rng'])
    loaded={**value,'model_shards':{n:local_parameter(p) for n,p in model.named_parameters()},
        'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state()}
    if digest(loaded)!=note['state_digest']:raise AssertionError('Exact local model/optimizer/RNG restoration failed')
    dist.barrier()
    return dict(update=update,cursor=value['cursor'],real_targets=value['real_targets'],
        neural_ops=value['neural_ops'],optimizer_ops=value['optimizer_ops'])


def reconstruct_weights(folder,update,config):
    folder=Path(folder);manifest=json.loads((folder/f'manifest-u{update}.json').read_text())
    if manifest['purpose']!=PURPOSE or manifest['eligible_for_science'] is not False or not manifest['all_replicated_states_equal']:
        raise ValueError('Complete numerical-only shard collection required')
    rows=[]
    for rank,note in enumerate(manifest['shards']):
        file=folder/note['file']
        if note['rank']!=rank or file_sha256(file)!=note['sha256']:raise AssertionError('Checkpoint shard fingerprint changed')
        saved=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
        validate_shard(saved,manifest['signature'],rank,manifest['world'],update)
        if digest(saved)!=note['state_digest']:raise AssertionError('Checkpoint shard contents changed')
        rows.append(saved)
    result={};world=manifest['world']
    for name in rows[0]['model_shards']:
        placement=rows[0]['optimizer']['schema']['placements'][name]
        tensors=[r['model_shards'][name] for r in rows]
        if placement=='R':
            if any(not torch.equal(tensors[0],t) for t in tensors[1:]):raise AssertionError('Replicated tensor changed')
            result[name]=tensors[0].clone()
        elif placement=='S(0)':result[name]=torch.cat(tensors,0)
        else:raise ValueError('Unexpected sharding placement')
    return result,rows
