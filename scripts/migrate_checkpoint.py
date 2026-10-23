"""Create a derived checkpoint without changing any model/Adam/RNG tensor.

The source remains immutable. Both global microbatches, source cursor, past
exposure and past FLOPs must be preserved. Implementation flags and a future
stable/decay schedule are explicit, auditable metadata changes.
"""
import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
from compute_trainium.checkpoint_migration import migration_signature, optimizer_implementation
from compute_trainium.continuation import validate_resume_signature, validate_resume_schedule
from compute_trainium.training import PackedCorpus, WSDSchedule, atomic_checkpoint, file_sha256
from scripts.qualify_native import digest


def migrate(parent, packed, metadata, corpus_sha, *, world, batch, accumulation,
            optimizer, decay_start, decay_end):
    expected=digest({'model':parent['model'],'optimizer':parent['optimizer']})
    if expected!=parent['state_digest']:
        raise ValueError('Source full-state digest differs')
    data_signature=dict(parent['signature'],corpus_sha256=corpus_sha,
                        packing_sha256=packed.fingerprint)
    validate_resume_signature(parent['signature'],data_signature,allow_extension=True,
        metadata=metadata,packed=packed,cursor=parent['cursor'],exposure=parent['real_tokens'])
    schedule=asdict(WSDSchedule(parent['schedule']['peak'],parent['schedule']['warmup_tokens'],
                               decay_start,decay_end,parent['schedule']['floor_ratio']))
    validate_resume_schedule(parent['schedule'],schedule,parent['real_tokens'],extend_stable=True)
    # Shallow-copy the large trees; the flags helper only copies optimizer metadata.
    result=dict(parent)
    result['signature']=migration_signature(parent['signature'],world=world,batch=batch,
        accumulation=accumulation,corpus_sha=corpus_sha,packing_sha=packed.fingerprint,optimizer=optimizer)
    result['optimizer']=optimizer_implementation(parent['optimizer'],optimizer)
    result['schedule']=schedule
    before=digest({'model':parent['model'],'moments':parent['optimizer']['state'],'rng':parent['cpu_rng']})
    after=digest({'model':result['model'],'moments':result['optimizer']['state'],'rng':result['cpu_rng']})
    if before!=after:raise AssertionError('Migration changed a scientific-state tensor')
    for key in ['cursor','update','real_tokens','training_matmul_flops','initial_model_digest','pauses']:
        if result[key]!=parent[key]:raise AssertionError('Changed past history: '+key)
    result['state_digest']=digest({'model':result['model'],'optimizer':result['optimizer']})
    proof={'status':'PASS_EXPLICIT_CHECKPOINT_MIGRATION','parent_state_digest':expected,
        'derived_state_digest':result['state_digest'],'unchanged_model_moments_rng_digest':before,
        'old_signature':parent['signature'],'new_signature':result['signature'],
        'old_schedule':parent['schedule'],'new_schedule':schedule,
        'cursor':result['cursor'],'real_tokens':result['real_tokens'],'update':result['update'],
        'training_matmul_flops':result['training_matmul_flops'],'global_microbatches_preserved':True,
        'original_packed_prefix_verified':True,'model_optimizer_moment_rng_tensors_unchanged':True}
    return result,proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--world',type=int,default=32);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--optimizer',choices=['plain','foreach','fused'],required=True)
    p.add_argument('--decay-start-update',type=int,required=True);p.add_argument('--end-update',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    if args.output.exists():raise FileExistsError(args.output)
    parent=torch.load(args.checkpoint,weights_only=False,mmap=True,map_location='cpu')
    source_sha=file_sha256(args.data/'source.json')
    packed=PackedCorpus.from_mapped(args.data/'packed',parent['signature']['context'],
                                   parent['signature']['data_seed'],source_sha)
    global_batch=args.world*args.batch*args.accumulation
    decay_start=int(packed.prefix_counts[args.decay_start_update*global_batch].sum())
    decay_end=int(packed.prefix_counts[args.end_update*global_batch].sum())
    result,proof=migrate(parent,packed,json.loads((args.data/'packed/manifest.json').read_text()),source_sha,
        world=args.world,batch=args.batch,accumulation=args.accumulation,optimizer=args.optimizer,
        decay_start=decay_start,decay_end=decay_end)
    proof['parent_checkpoint_sha256']=file_sha256(args.checkpoint)
    atomic_checkpoint(args.output,result)
    restored=torch.load(args.output,weights_only=False,mmap=True,map_location='cpu')
    if digest({'model':restored['model'],'optimizer':restored['optimizer']})!=result['state_digest']:
        raise AssertionError('Written derived state did not recover exactly')
    proof['derived_checkpoint_sha256']=file_sha256(args.output)
    args.output.with_suffix('.migration.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if not isinstance(v,dict)}),flush=True)


if __name__=='__main__':main()
