"""Bounded private evidence for rejected replicas; never a resumable state.

The original exact replica digest remains the admission criterion. This module
does no work on an agreeing checkpoint and cannot select a majority to resume.
Native adoption requires a separately frozen, qualified source bundle.
"""
import hashlib
import json
from pathlib import Path
import shutil

import torch
from torch.utils._pytree import tree_flatten_with_path, keystr

from .training import atomic_checkpoint, cpu_tree, file_sha256

PURPOSE = 'rejected_replica_diagnostic_only'
GIB = 1024**3
MAX_STATE_BYTES = 3*GIB
MAX_SELECTED_RANKS = 2
MAX_CONCURRENT_GROUPS = 2  # Two independent eight-chip workers on this instance.


def selected_ranks(digests):
    if (not isinstance(digests, list) or not 2 <= len(digests) <= 64
            or any(not isinstance(v, str) or len(v) != 64
                   or any(c not in '0123456789abcdef' for c in v) for v in digests)):
        raise ValueError('Bounded exact SHA256 replica inventory required')
    other = next((i for i, value in enumerate(digests) if value != digests[0]), None)
    return () if other is None else (0, other)


def tensor_bytes(state):
    leaves, _ = tree_flatten_with_path(state)
    if len(leaves) > 50000:
        raise ValueError('Diagnostic leaf inventory exceeds bound')
    metadata = 0
    for path, value in leaves:
        name = keystr(path)
        if len(name) > 65536:
            raise ValueError('Diagnostic path exceeds metadata bound')
        metadata += len(name.encode())
        if not isinstance(value, torch.Tensor):
            if not isinstance(value, (str, int, float, bool, type(None), torch.dtype, torch.device)):
                raise ValueError('Unsupported diagnostic metadata leaf')
            if isinstance(value, str) and len(value) > 1024**2:
                raise ValueError('Diagnostic metadata string exceeds bound')
            metadata += len(repr(value).encode())
        if metadata > 16*1024**2:
            raise ValueError('Diagnostic metadata exceeds bound')
    return sum(v.numel()*v.element_size() for _, v in leaves if isinstance(v, torch.Tensor))


def _available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('Host memory reserve cannot be verified')


def _cgroup_headroom():
    root = Path('/sys/fs/cgroup')
    maximum, current = root/'memory.max', root/'memory.current'
    if not maximum.exists() or not current.exists():
        return None  # Host admission still requires its full protected reserve.
    value = maximum.read_text().strip()
    return None if value == 'max' else int(value)-int(current.read_text())


def require_diagnostic_reserves(directory, size):
    if not 0 <= size <= MAX_STATE_BYTES:
        raise ValueError('Rejected state exceeds three-GiB per-rank bound')
    # Both eight-chip groups can reject together: four private states globally.
    disk = MAX_CONCURRENT_GROUPS*MAX_SELECTED_RANKS*(MAX_STATE_BYTES+64*1024**2)
    memory = MAX_CONCURRENT_GROUPS*MAX_SELECTED_RANKS*MAX_STATE_BYTES+GIB
    if shutil.disk_usage(directory).free < 35*GIB+disk:
        raise RuntimeError('Private evidence would breach protected disk reserve')
    if _available_memory() < 512*GIB+memory:
        raise RuntimeError('Private evidence would breach protected host reserve')
    headroom = _cgroup_headroom()
    container_memory = MAX_SELECTED_RANKS*MAX_STATE_BYTES+GIB
    if headroom is not None and headroom < container_memory:
        raise RuntimeError('Insufficient diagnostic container memory headroom')


def leaf_inventory(state):
    leaves, spec = tree_flatten_with_path(state)
    result = []
    for path, value in leaves:
        row = {'path': keystr(path)}
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous()
            row.update(kind='tensor', dtype=str(array.dtype), shape=list(array.shape),
                sha256=hashlib.sha256(array.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        else:
            row.update(kind='metadata', type=type(value).__qualname__,
                sha256=hashlib.sha256(repr(value).encode()).hexdigest())
        result.append(row)
    return {'structure_sha256': hashlib.sha256(repr(spec).encode()).hexdigest(), 'leaves': result}


def capture_rejected(directory, *, update, rank, digests, state_factory, digest_fn):
    """At most rank0 and the first disagreeing rank; all callers must still fail.

    Evidence is best effort under protected reserves. No state factory is called
    for agreement or an unselected rank. A receipt records capture failures, and
    the scientific caller must reject regardless of this function's status.
    """
    chosen = selected_ranks(digests)
    if type(rank) is not int or not 0 <= rank < len(digests) or type(update) is not int or update < 1:
        raise ValueError('Exact rank and positive completed update required')
    if not chosen:
        return {'status':'AGREEMENT_NO_DIAGNOSTIC', 'selected_ranks':[]}
    if rank not in chosen:
        return {'status':'UNSELECTED_REJECTED_REPLICA', 'selected_ranks':list(chosen)}
    directory = Path(directory)
    receipt = {'status':'FAILED_REJECTED_STATE_CAPTURE', 'rank':rank, 'update':update,
               'selected_ranks':list(chosen), 'eligible_for_resume':False,
               'recorded_state_digest':digests[rank]}
    stem = f'rejected-u{update}-rank{rank}'
    try:
        file, note = directory/(stem+'.pt'), directory/(stem+'.json')
        if file.exists() or note.exists() or file.with_suffix('.tmp').exists():
            raise FileExistsError('Preserve earlier rejected-state evidence')
        state = state_factory()
        size = tensor_bytes(state)
        require_diagnostic_reserves(directory, size)
        state = cpu_tree(state)
        snapshot_digest = digest_fn(state)
        payload = {'purpose':PURPOSE, 'qualification_only':True, 'eligible_for_resume':False,
            'rank':rank, 'update':update, 'recorded_state_digest':digests[rank],
            'saved_state_digest':snapshot_digest, 'rejected_complete_state':state}
        atomic_checkpoint(file, payload)
        saved = torch.load(file, map_location='cpu', mmap=True, weights_only=False)
        if digest_fn(saved['rejected_complete_state']) != snapshot_digest:
            raise AssertionError('Rejected-state disk readback changed')
        receipt.update(status='CAPTURED_PRIVATE_REJECTED_STATE', snapshot_file=file.name,
            snapshot_sha256=file_sha256(file), snapshot_bytes=file.stat().st_size,
            tensor_bytes=size, saved_state_digest=snapshot_digest,
            repeated_readback_matches_original_digest=snapshot_digest == digests[rank],
            inventory=leaf_inventory(saved['rejected_complete_state']))
    except Exception as error:
        receipt.update(error_kind=type(error).__name__, error=str(error))
    try:
        # Exclusive creation preserves earlier receipts, including capture failure.
        with (directory/(stem+'.json')).open('x') as handle:
            json.dump(receipt, handle, indent=2); handle.write('\n')
    except OSError:
        pass  # The original scientific replica assertion must remain the failure.
    return receipt


def enforce_replica_agreement(directory, *, update, rank, digests, state_factory,
                              digest_fn, barrier, failure_message):
    """Preserve bounded evidence, synchronize, then reject on every rank."""
    if not selected_ranks(digests):
        return
    receipt = capture_rejected(directory, update=update, rank=rank, digests=digests,
                               state_factory=state_factory, digest_fn=digest_fn)
    barrier()
    raise AssertionError(f'{failure_message}; rejected-state evidence: {receipt["status"]}')


def compare_inventories(left, right):
    """Localize leaves without publishing raw weights, moments or tensor arrays."""
    l = {v['path']:v for v in left['leaves']}
    r = {v['path']:v for v in right['leaves']}
    if len(l) != len(left['leaves']) or len(r) != len(right['leaves']):
        raise ValueError('Ambiguous diagnostic leaf paths')
    changed = [p for p in sorted(l.keys() & r.keys()) if l[p] != r[p]]
    return {'changed_leaf_paths':changed, 'left_only':sorted(l.keys()-r.keys()),
        'right_only':sorted(r.keys()-l.keys()),
        'structure_equal':left['structure_sha256'] == right['structure_sha256'],
        'all_leaf_values_equal':not changed and l.keys() == r.keys(),
        'scope':'Leaf hashes localize differences;they do not quantify numerical error or certify training.'}
