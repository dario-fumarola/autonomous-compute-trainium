"""Explicit checks for future schedule extension and append-only source data."""
import hashlib


def prefix_fingerprint(tokens, domains, length):
    value=hashlib.sha256()
    for array in (tokens,domains):
        for first in range(0,length,1024*1024):
            value.update(array[first:min(length,first+1024*1024)].numpy().tobytes())
    return value.hexdigest()


def validate_resume_signature(old,new,*,allow_extension=False,metadata=None,packed=None,cursor=0,exposure=0):
    if old==new:return
    if not allow_extension or metadata is None or packed is None:
        raise ValueError('Checkpoint configuration/data fingerprint mismatch')
    omit={'corpus_sha256','packing_sha256'}
    if {k:v for k,v in old.items() if k not in omit}!={k:v for k,v in new.items() if k not in omit}:
        raise ValueError('Data extension must preserve every model/optimizer/topology setting')
    parent=metadata.get('prefix_parent',{})
    if (parent.get('fingerprint')!=old['packing_sha256'] or parent.get('source_sha256')!=old['corpus_sha256']
        or cursor>parent.get('windows',0)):
        raise ValueError('Extension is not a descendant of the recovered source')
    length=parent['windows']*new['context']+1
    if prefix_fingerprint(packed.tokens,packed.domains,length)!=old['packing_sha256']:
        raise ValueError('Extended data changed the original packed prefix')
    if int(packed.prefix_counts[cursor].sum())!=exposure:
        raise ValueError('Recovered exposure and extended prefix counts disagree')


def validate_resume_schedule(old,new,exposure,*,decay_branch=False,extend_stable=False):
    if old==new:return
    a=dict(old,decay_start=None,decay_end=None)
    b=dict(new,decay_start=None,decay_end=None)
    if a!=b:raise ValueError('Continuation must preserve peak/warmup/floor')
    if decay_branch and old['decay_start'] is None and new['decay_start']==exposure:return
    if extend_stable:
        if old['decay_start'] is not None and exposure>old['decay_start']:
            raise ValueError('Cannot extend a checkpoint after annealing has begun')
        if new['decay_start'] is not None and new['decay_start']<exposure:
            raise ValueError('New decay cannot rewrite the completed stable history')
        return
    raise ValueError('Schedule change requires an explicit stable extension or decay branch')
