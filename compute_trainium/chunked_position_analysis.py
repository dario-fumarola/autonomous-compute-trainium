"""Fixed position strata; source-level pairing, with no outcome-based slicing."""
import math

import numpy as np

from scripts.analyze_probability_screen import paired, score

STRATA = ('before_first_slot', 'slot_prediction', 'after_1_to_8',
          'after_9_to_32', 'after_33_to_127')
DOMAINS = ('web', 'math', 'code')
WEIGHTS = (.6, .2, .2)


def position_strata(length=512, stride=128):
    if length != 512 or stride != 128:
        raise ValueError('This diagnostic supports only the frozen 512/128 design')
    rows = {s: [] for s in STRATA}
    for i in range(length):
        offset = (i + 1) % stride
        if offset == 0:
            key = 'slot_prediction'
        elif i < stride - 1:
            key = 'before_first_slot'
        elif offset <= 8:
            key = 'after_1_to_8'
        elif offset <= 32:
            key = 'after_9_to_32'
        else:
            key = 'after_33_to_127'
        rows[key].append(i)
    return rows


def make_rows(losses, metadata, indices):
    values = np.asarray(losses, dtype=np.float64)
    if values.shape != (len(indices), 512) or not np.isfinite(values).all() or (values < 0).any():
        raise AssertionError('Finite per-target losses on the complete frozen context required')
    if len(set(indices)) != len(indices):
        raise AssertionError('Repeated evaluation indices')
    bins = position_strata()
    output = []
    for j, i in enumerate(indices):
        m = metadata[i]
        output.append(dict(index=i, id=m['id'], group=m['group'], domain=m['domain'],
            targets=512, loss_sum=math.fsum(values[j]),
            strata={s: dict(targets=len(positions), loss_sum=math.fsum(values[j, positions]))
                    for s, positions in bins.items()}))
    return output


def validate(rows):
    if len(rows) != 384 or {r['index'] for r in rows} != set(range(384)) or len({r['id'] for r in rows}) != 384:
        raise AssertionError('Complete unique original panel required')
    if any(sum(r['domain'] == d for r in rows) != 128 for d in DOMAINS):
        raise AssertionError('Frozen domain balance changed')
    sizes = {s: len(p) for s, p in position_strata().items()}
    for r in rows:
        if r['targets'] != 512 or set(r['strata']) != set(STRATA):
            raise AssertionError('Full target coverage required')
        if any(r['strata'][s]['targets'] != sizes[s] for s in STRATA):
            raise AssertionError('Every target appears in exactly one fixed stratum')
        if not math.isclose(math.fsum(r['strata'][s]['loss_sum'] for s in STRATA),
                            r['loss_sum'], abs_tol=1e-9, rel_tol=1e-12):
            raise AssertionError('Strata must reconstruct full loss')
        if any(not math.isfinite(v['loss_sum']) or v['loss_sum'] < 0 for v in [r, *r['strata'].values()]):
            raise AssertionError('Non-finite or negative loss')


def summarize(rows):
    validate(rows)
    domains = {d: score([r for r in rows if r['domain'] == d]) for d in DOMAINS}
    return dict(domains=domains, broad_nll=math.fsum(w*domains[d] for d, w in zip(DOMAINS, WEIGHTS)))


def compare(candidate, control, seed=40514, replicates=2000):
    validate(candidate); validate(control)
    left = {r['id']: r for r in control}
    right = {r['id']: r for r in candidate}
    if left.keys() != right.keys() or any(
        tuple(left[k][f] for f in ('index', 'group', 'domain')) !=
        tuple(right[k][f] for f in ('index', 'group', 'domain')) for k in left):
        raise AssertionError('Original paired source metadata required')
    def pair(a, b):
        return dict(broad=paired(a, b, seed=seed, replicates=replicates, weighted=True),
            domains={d: paired([r for r in a if r['domain'] == d],
                               [r for r in b if r['domain'] == d],
                               seed=seed, replicates=replicates) for d in DOMAINS})
    result = dict(full=pair(control, candidate), strata={})
    for s, positions in position_strata().items():
        project = lambda rows: [{**r, **r['strata'][s]} for r in rows]
        value = pair(project(control), project(candidate))
        value['target_fraction'] = len(positions)/512
        value['contribution_to_broad_gain'] = value['broad']['gain']*len(positions)/512
        result['strata'][s] = value
    if not math.isclose(math.fsum(v['contribution_to_broad_gain'] for v in result['strata'].values()),
                        result['full']['broad']['gain'], abs_tol=1e-12, rel_tol=1e-10):
        raise AssertionError('Position contributions must reconstruct the broad gain')
    return result
