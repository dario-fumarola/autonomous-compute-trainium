"""Explicit metadata migration, with immutable tensor state and global batches."""
import copy


def migration_signature(old,*,world,batch,accumulation,corpus_sha,packing_sha,optimizer):
    before=(old['world']*old['batch'],old['accumulation'])
    after=(world*batch,accumulation)
    if before!=after:
        raise ValueError('Migration must preserve every global microbatch and accumulated update')
    new=copy.deepcopy(old)
    new.update(world=world,batch=batch,accumulation=accumulation,
               corpus_sha256=corpus_sha,packing_sha256=packing_sha,optimizer=optimizer)
    return new


def optimizer_implementation(state,mode):
    if mode not in ['plain','foreach','fused']:raise ValueError('Unknown optimizer implementation')
    result=copy.deepcopy(state)
    for group in result['param_groups']:
        group['foreach']=mode=='foreach';group['fused']=mode=='fused'
    return result
