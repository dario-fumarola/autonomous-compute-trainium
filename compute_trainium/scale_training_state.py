"""Same-topology training state, separate from all qualification lineages.

The numerical optimizer's tensor format is an internal execution format. Its
wrapper never turns qualification weights into an eligible training lineage.
"""
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist

from .scale_model_sharding import StreamingOwnerOptimizer
from .scale_model_state import replicated_state
from .scale_sharded_candidate import local_parameter
from .scale_sharded_fixture import cpu_tree
from .training import atomic_checkpoint,file_sha256
from scripts.qualify_native import digest

PURPOSE='scale_registered_training_state_v1'


def validate_lineage(signature):
    required={'study_sha256','case','phase','initializer_sha256','packing_fingerprint','model_role','state_seed'}
    if not required<=signature.keys() or signature['phase'] not in ('gate','tune','main'):
        raise ValueError('Complete frozen training lineage required')
    if signature['model_role'] not in ('fixed_two','ordinary8'):
        raise ValueError('Qualified model role required')


class ScaleTrainingOptimizer(StreamingOwnerOptimizer):
    """Unchanged qualified update math with a distinct training state wrapper."""
    def __init__(self,*args,lineage,**kwargs):
        validate_lineage(lineage);self.lineage=copy.deepcopy(lineage)
        super().__init__(*args,**kwargs)

    def state_dict(self):
        return dict(purpose=PURPOSE,lineage=copy.deepcopy(self.lineage),execution=super().state_dict())

    def load_state_dict(self,value):
        if value.get('purpose')!=PURPOSE or value.get('lineage')!=self.lineage:
            raise ValueError('Qualification and other training lineages cannot be imported')
        super().load_state_dict(value['execution'])


def validate_saved(value,signature,rank,world,update,per_update):
    validate_lineage(signature)
    eligible=signature['phase']!='gate'
    if (value.get('purpose')!=PURPOSE or value.get('signature')!=signature
            or value.get('eligible_for_science')!=eligible
            or value.get('qualification_only')!=(not eligible)
            or value.get('rank')!=rank or value.get('world')!=world
            or value.get('update')!=update or value.get('cursor')!=update*512
            or value.get('work')!=update*per_update
            or value.get('optimizer',{}).get('lineage')!=signature
            or value.get('optimizer',{}).get('execution',{}).get('update')!=update):
        raise ValueError('Exact lineage, topology, data cursor and Python-integer work required')


def save_training_shard(model,optimizer,folder,signature,update,packed,per_update):
    validate_lineage(signature)
    if optimizer.update!=update:raise AssertionError('Optimizer and source update counters differ before save')
    rank,world=dist.get_rank(),dist.get_world_size();folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    value=dict(purpose=PURPOSE,qualification_only=signature['phase']=='gate',
        eligible_for_science=signature['phase']!='gate',signature=signature,
        rank=rank,world=world,update=update,cursor=update*512,
        domain_targets=list(map(int,packed.counts(0,update*512))) if update else [0,0,0],
        work=update*per_update,model_shards={n:local_parameter(p) for n,p in model.named_parameters()},
        optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state())
    cpu=cpu_tree(value);state_digest=digest(cpu);file=folder/f'rank{rank}-u{update}.pt'
    if file.exists():raise FileExistsError('Training checkpoint cannot overwrite retained evidence')
    atomic_checkpoint(file,cpu)
    if digest(torch.load(file,map_location='cpu',mmap=True,weights_only=False))!=state_digest:
        raise AssertionError('Training shard disk readback differs')
    note=dict(rank=rank,file=file.name,sha256=file_sha256(file),state_digest=state_digest,
        update=update,cursor=update*512,domain_targets=value['domain_targets'],work=value['work'],
        replicated_state_digest=digest(replicated_state(optimizer)),schema=optimizer.schema)
    (folder/f'rank{rank}-u{update}.json').write_text(json.dumps(note,indent=2)+'\n');dist.barrier()
    if rank==0:
        notes=[json.loads((folder/f'rank{i}-u{update}.json').read_text()) for i in range(world)]
        if len({n['replicated_state_digest'] for n in notes})!=1:
            raise AssertionError('Replica discrepancy; every raw shard retained, no majority state selected')
        if any(any(n[k]!=note[k] for k in ['schema','update','cursor','domain_targets','work']) for n in notes):
            raise AssertionError('Common scientific state/counters differ')
        manifest=dict(purpose=PURPOSE,signature=signature,world=world,update=update,
            eligible_for_science=value['eligible_for_science'],all_replicated_states_equal=True,shards=notes)
        (folder/f'manifest-u{update}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    dist.barrier();return state_digest


def load_training_shard(model,optimizer,folder,signature,update,packed,per_update):
    rank,world=dist.get_rank(),dist.get_world_size();folder=Path(folder)
    manifest=json.loads((folder/f'manifest-u{update}.json').read_text())
    if (manifest.get('purpose')!=PURPOSE or manifest.get('signature')!=signature
            or manifest.get('world')!=world or manifest.get('update')!=update
            or not manifest.get('all_replicated_states_equal')):
        raise ValueError('Complete exact training manifest required')
    note=manifest['shards'][rank];file=folder/note['file']
    if note['rank']!=rank or file_sha256(file)!=note['sha256']:raise AssertionError('Training shard changed')
    value=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
    validate_saved(value,signature,rank,world,update,per_update)
    counts=list(map(int,packed.counts(0,update*512))) if update else [0,0,0]
    if value['domain_targets']!=counts or digest(value)!=note['state_digest']:
        raise AssertionError('Training data/counters/contents differ')
    device=local_parameter(next(model.parameters())).device
    with torch.no_grad():
        for name,p in model.named_parameters():local_parameter(p).copy_(value['model_shards'][name].to(device))
    optimizer.load_state_dict(value['optimizer']);torch.set_rng_state(value['cpu_rng'])
    loaded={**value,'model_shards':{n:local_parameter(p) for n,p in model.named_parameters()},
        'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state()}
    if digest(loaded)!=note['state_digest']:raise AssertionError('Training model/optimizer/RNG restore differs')
    dist.barrier();return value['update']


def reconstruct_training_weights(folder,update,signature,per_update):
    folder=Path(folder);manifest=json.loads((folder/f'manifest-u{update}.json').read_text());rows=[]
    if (manifest.get('purpose')!=PURPOSE or manifest.get('signature')!=signature
            or not manifest.get('all_replicated_states_equal')):raise ValueError('Complete training lineage required')
    for rank,note in enumerate(manifest['shards']):
        file=folder/note['file']
        if note['rank']!=rank or file_sha256(file)!=note['sha256']:raise AssertionError('Training shard changed')
        value=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
        validate_saved(value,signature,rank,manifest['world'],update,per_update)
        if digest(value)!=note['state_digest']:raise AssertionError('Training state digest differs')
        rows.append(value)
    result={};schema=rows[0]['optimizer']['execution']['schema']
    for name in rows[0]['model_shards']:
        tensors=[r['model_shards'][name] for r in rows]
        if schema['placements'][name]=='R':
            if any(not torch.equal(tensors[0],t) for t in tensors[1:]):raise AssertionError('Replica differs')
            result[name]=tensors[0].clone()
        elif schema['placements'][name]=='S(0)':result[name]=torch.cat(tensors,0)
        else:raise ValueError('Unknown training placement')
    return result,rows


def verify_live_training_state(model,optimizer,folder,signature,update,per_update):
    """Detect mutation by a zero-update diagnostic; never restore over it."""
    rank,world=dist.get_rank(),dist.get_world_size();folder=Path(folder)
    note=json.loads((folder/f'rank{rank}-u{update}.json').read_text())
    saved=torch.load(folder/note['file'],map_location='cpu',mmap=True,weights_only=False)
    validate_saved(saved,signature,rank,world,update,per_update)
    live={**saved,'model_shards':{n:local_parameter(p) for n,p in model.named_parameters()},
        'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state()}
    if digest(live)!=note['state_digest']:raise AssertionError('Zero-update diagnostic mutated live model/optimizer/RNG')
    dist.barrier()
