"""Align an independent optimizer by names while preserving its group policy."""
from collections import Counter


def align_reference_groups(model,groups,target_names):
    named=dict(model.named_parameters())
    lookup={id(p):name for name,p in named.items()}
    original=[[lookup[id(p)] for p in g['params']] for g in groups]
    flattened=[name for group in target_names for name in group]
    if len(flattened)!=len(set(flattened)) or set(flattened)!=set(named):
        raise ValueError('Target optimizer duplicated or omitted a named parameter')
    # Equality of complete group membership, not just flattened inventory:
    # norms/embeddings must remain AdamW and each Muon shape group unchanged.
    if Counter(map(frozenset,original))!=Counter(map(frozenset,target_names)):
        raise ValueError('Independent optimizer group membership differs')
    by_membership={frozenset(names):g for names,g in zip(original,groups,strict=True)}
    result=[]
    for index,names in enumerate(target_names):
        group=by_membership[frozenset(names)]
        if group['kind']!=('adamw' if index==0 else 'muon'):
            raise ValueError('Independent optimizer family changed')
        if index and len({tuple(named[n].shape) for n in names})!=1:
            raise ValueError('Muon group matrix shapes differ')
        result.append({**group,'params':[named[n] for n in names]})
    return result
