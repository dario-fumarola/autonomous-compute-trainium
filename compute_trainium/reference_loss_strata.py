"""Descriptive position losses in bins fixed by an independent reference."""
import numpy as np


def reference_deciles(reference,baseline,candidate,index,*,weights=None,bins=10):
    weights=weights or {'web':.6,'math':.2,'code':.2}
    arrays=[np.asarray(v,dtype=np.float64) for v in [reference,baseline,candidate]]
    if not 1<=bins<=10 or any(v.ndim!=1 or v.shape!=arrays[0].shape or not np.isfinite(v).all() for v in arrays):raise ValueError('Finite exactly paired position-loss arrays required')
    if set(weights)!={r['domain'] for r in index} or abs(sum(weights.values())-1)>1e-12:raise ValueError('Declared domain weights/coverage differ')
    positions={d:[] for d in weights};offset=0;seen=set()
    for row in index:
        if row['id'] in seen or row['offset']!=offset or type(row['length']) is not int or row['length']<1:raise ValueError('Unique contiguous complete source positions required')
        seen.add(row['id']);positions[row['domain']].extend(range(offset,offset+row['length']));offset+=row['length']
    if offset!=len(arrays[0]):raise ValueError('Position coverage differs from the original targets')
    rows=[];recovered=0.
    for domain,raw in positions.items():
        ids=np.asarray(raw,dtype=np.int64)
        if len(ids)<bins:raise ValueError('Too few original positions for the fixed bins')
        ordered=ids[np.lexsort((ids,arrays[0][ids]))]
        for number,selected in enumerate(np.array_split(ordered,bins),1):
            gain=float((arrays[1][selected]-arrays[2][selected]).mean())
            rows.append(dict(domain=domain,bin=number,targets=len(selected),reference_mean_nll=float(arrays[0][selected].mean()),baseline_mean_nll=float(arrays[1][selected].mean()),candidate_mean_nll=float(arrays[2][selected].mean()),gain_nats=gain))
            recovered+=weights[domain]*len(selected)/len(ids)*gain
    pooled=sum(weights[d]*float((arrays[1][np.asarray(ids)]-arrays[2][np.asarray(ids)]).mean()) for d,ids in positions.items())
    if abs(recovered-pooled)>1e-10:raise AssertionError('Stratification changed the fixed pooled outcome')
    return dict(rows=rows,pooled_gain_nats=pooled,reconstructed_pooled_gain_nats=recovered,scope='Secondary descriptive bins within each domain,ranked by independent frozen reference loss;tied losses use original document/position order. Target-dependent diagnostic,not a deployable routing score. No token-IID significance or replacement of pooled NLL.')
