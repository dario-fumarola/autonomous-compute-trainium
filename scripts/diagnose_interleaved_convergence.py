"""Frozen TRAIN-only numerical-depth and sequential consistency diagnostic."""
import argparse
import json
import resource
import shutil
import time
from pathlib import Path

ID = 'interleaved-convergence-v1'
PUBLIC = Path('results/interleaved_workspace_20261029/convergence-v1')
TUNING = 'interleaved-lr-tuning-v1'
ROUNDS = [0, 1, 2, 3, 4, 8, 16, 32]


def write(path, value):
    from compute_trainium.queue import write_state
    write_state(path, value)


def prepare(root):
    import hashlib
    import numpy as np
    from compute_trainium.indexed_natural_sources import IndexedNaturalSources, DOMAINS
    from compute_trainium.training import file_sha256
    tuning_path = root / TUNING / 'plan.json'
    tuning = json.loads(tuning_path.read_text())
    data = IndexedNaturalSources(root / tuning['data_relative'])
    start = tuning['audit_window_start'] + 128
    selected = {d: [] for d in DOMAINS}
    for i in range(start, min(start+20000, data.windows)):
        tokens, domains = data.window(i)
        unique = np.unique(domains[1:])
        if len(unique) != 1 or unique[0] == 0:
            continue
        domain = DOMAINS[int(unique[0])-1]
        if len(selected[domain]) < 2:
            selected[domain].append(dict(index=i, domain=domain,
                tokens_sha256=hashlib.sha256(tokens.astype('<i8').tobytes()).hexdigest()))
        if all(len(v) == 2 for v in selected.values()):
            break
    if any(len(v) != 2 for v in selected.values()):
        raise AssertionError('Registered source-only window selection exhausted')
    files = ['compute_trainium/interleaved_sequential_reference.py',
             'scripts/diagnose_interleaved_convergence.py',
             'docs/INTERLEAVED_CONVERGENCE_DIAGNOSTIC_20261029.md']
    for f in tuning['source_sha256']:
        if f.startswith('compute_trainium/'):
            files.append(f)
    now = time.time()
    p = dict(identity=ID, registered_unix=now, absolute_deadline_unix=now+5400,
        endpoint_seconds_limit=1200, rss_limit_bytes=16*1024**3, threads=16,
        tuning_plan_sha256=file_sha256(tuning_path),
        source_sha256={f: file_sha256(f) for f in files},
        windows=[row for domain in DOMAINS for row in selected[domain]],
        arms=['initialization', 'latent-lr0', 'latent-lr1', 'latent-lr2'], rounds=ROUNDS,
        guard=dict(maximum_absolute_domain_nll_shift=.01, maximum_domain_kl=.01,
                   maximum_domain_k16_k32_kl=.0001),
        assigned_chips=0, optimizer_updates=0, heldout_reads=0,
        scope='TRAIN-only fixed-round/sequential consistency, not learning-rate selection, held-out quality, reasoning or serving speed.')
    (root / ID).mkdir(); PUBLIC.mkdir()
    write(root / ID / 'plan.json', p); write(PUBLIC / 'plan.json', p)
    print(json.dumps(dict(status='REGISTERED_CONVERGENCE_DIAGNOSTIC', windows=len(p['windows']))))


def run(root):
    import hashlib
    import torch
    from torch.nn import functional as F
    from compute_trainium.indexed_natural_sources import IndexedNaturalSources, DOMAINS
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.interleaved_sequential_reference import cached_sequential
    from compute_trainium.training import file_sha256
    from scripts.qualify_native import digest
    out = root / ID; p = json.loads((out / 'plan.json').read_text())
    if any(file_sha256(f) != h for f, h in p['source_sha256'].items()):
        raise AssertionError('Frozen diagnostic source changed')
    tuning_path = root / TUNING / 'plan.json'
    if file_sha256(tuning_path) != p['tuning_plan_sha256']:
        raise AssertionError('Frozen training plan changed')
    tuning = json.loads(tuning_path.read_text())
    data = IndexedNaturalSources(root / tuning['data_relative'])
    torch.set_num_threads(p['threads'])
    start = time.time(); result = dict(status='RUNNING_CONVERGENCE_DIAGNOSTIC', arms={})
    def guard(endpoint_start=None):
        if time.time() > p['absolute_deadline_unix']:
            raise RuntimeError('Absolute diagnostic deadline')
        if endpoint_start is not None and time.time()-endpoint_start > p['endpoint_seconds_limit']:
            raise RuntimeError('Endpoint CPU time cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > p['rss_limit_bytes']:
            raise RuntimeError('Diagnostic RSS cap')
        mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        if mem['MemAvailable'] < 512*1024**3 or min(shutil.disk_usage(root).free, shutil.disk_usage('/').free) < 35*1024**3:
            raise RuntimeError('Protected host/disk reserve')
    def logs(logits):
        return logits.double().log_softmax(-1)
    def nll(lp, target):
        return -lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    def kl(a, b):
        return float((a.exp()*(a-b)).sum(-1).mean())
    try:
        for arm in p['arms']:
            audit = None
            if arm != 'initialization':
                audit_path = root / TUNING / ('audit-'+arm) / 'report.json'
                while not audit_path.exists():
                    guard(); time.sleep(10)
                audit = json.loads(audit_path.read_text())
                if audit['status'] != 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':
                    raise AssertionError('Complete independent audit required')
            endpoint_start = time.time(); guard(endpoint_start)
            torch.manual_seed(tuning['seed']); model = make_model(tuning).eval()
            record = dict(rows=[], initial_model_digest=digest(model.state_dict()))
            if audit:
                a = tuning['arms'][arm]; update = a['checkpoint_updates'][-1]
                checkpoint = root / 'runs' / a['phase'] / f'PRIVATE-u{update}.pt'
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
                if saved['initial_digest'] != record['initial_model_digest']:
                    raise AssertionError('Paired initialization differs')
                expected = audit['rows'][-1]['checkpoint_sha256']
                if file_sha256(checkpoint) != expected:
                    raise AssertionError('Audited checkpoint bytes changed')
                model.load_state_dict(saved['model']); del saved
                record.update(checkpoint_sha256=expected, audit_sha256=file_sha256(audit_path))
            else:
                # Bind this untouched initialization to the already completed
                # ordinary fit, not a newly selected random baseline.
                ordinary = root / 'runs' / tuning['arms']['ordinary-lr0']['phase'] / 'report.json'
                if ordinary.exists() and json.loads(ordinary.read_text())['initial_model_digest'] != record['initial_model_digest']:
                    raise AssertionError('Initial seed does not match Native fit')
            with torch.no_grad():
                for view in p['windows']:
                    guard(endpoint_start)
                    raw, _ = data.window(view['index'])
                    if hashlib.sha256(raw.astype('<i8').tobytes()).hexdigest() != view['tokens_sha256']:
                        raise AssertionError('Frozen TRAIN view changed')
                    x, y, mask = data.batch([view['index']])
                    if not bool(mask.all()):
                        raise AssertionError('Selected window must contain only real targets')
                    a, za = cached_sequential(model, x[:, :32], return_states=True)
                    b, zb = model.sequential(x[:, :32], return_states=True)
                    reference_error = float((nll(logs(a), y[:, :32])-nll(logs(b), y[:, :32])).abs().max())
                    if reference_error > .0002:
                        raise AssertionError('Cached sequential reference per-target numerical gate')
                    seq, zseq = cached_sequential(model, x, return_states=True)
                    seq_lp = logs(seq); del seq, a, b, za, zb
                    source = model.embeddings(x); pos = torch.arange(x.shape[1])
                    state = model.body_at(source, pos); selected = {}; rounds = []; k2_lp = None
                    for k in range(max(ROUNDS)+1):
                        guard(endpoint_start)
                        if k in ROUNDS:
                            final = model.body_at(model.interleave(source, state), pos.repeat_interleave(2))[:, 1::2]
                            lp = logs(F.linear(final, model.prediction_weight))
                            row = dict(k=k, nll=float(nll(lp, y).mean()), kl_to_sequential=kl(lp, seq_lp),
                                state_rmse_to_sequential=float((state-zseq).square().mean().sqrt()))
                            if k == 2:
                                k2_lp = lp; row['absolute_nll_shift_from_sequential'] = abs(row['nll']-float(nll(seq_lp, y).mean()))
                            if k2_lp is not None:
                                row['kl_from_k2'] = kl(k2_lp, lp)
                            if k == 16: selected[16] = lp
                            if k == 32: row['kl_from_k16'] = kl(selected[16], lp)
                            rounds.append(row)
                        if k < max(ROUNDS):
                            new = model.body_at(model.interleave(source, state), pos.repeat_interleave(2))[:, ::2]
                            # Store every adjacent update, including first unseen K3.
                            selected.setdefault('state_updates', []).append(dict(k=k+1, rmse=float((new-state).square().mean().sqrt())))
                            state = new
                    record['rows'].append(dict(index=view['index'], domain=view['domain'], real_targets=int(mask.sum()),
                        sequential_nll=float(nll(seq_lp, y).mean()), cached_reference_maximum_target_error=reference_error,
                        rounds=rounds, state_updates=selected['state_updates']))
                    write(out / ('progress-'+arm+'.json'), dict(completed_views=len(record['rows']), seconds=time.time()-endpoint_start))
                    del selected, seq_lp, zseq, k2_lp, lp, state, source
            domains = {}
            for domain in DOMAINS:
                rows = [r for r in record['rows'] if r['domain'] == domain]
                k2 = [next(v for v in r['rounds'] if v['k'] == 2) for r in rows]
                k32 = [next(v for v in r['rounds'] if v['k'] == 32) for r in rows]
                item = dict(absolute_mean_nll_shift=abs(sum(v['nll']-r['sequential_nll'] for v, r in zip(k2, rows))/len(rows)),
                    mean_kl_k2_to_sequential=sum(v['kl_to_sequential'] for v in k2)/len(rows),
                    mean_kl_k16_to_k32=sum(v['kl_from_k16'] for v in k32)/len(rows))
                item['eligible'] = (item['absolute_mean_nll_shift'] <= .01 and
                    item['mean_kl_k2_to_sequential'] <= .01 and item['mean_kl_k16_to_k32'] <= .0001)
                domains[domain] = item
            record.update(domains=domains, fixed_two_sequential_guard=all(v['eligible'] for v in domains.values()),
                wall_seconds=time.time()-endpoint_start)
            result['arms'][arm] = record; write(PUBLIC / (arm+'.json'), record)
            write(out / 'progress.json', result); del model
        result['status'] = 'COMPLETE_INTERLEAVED_CONVERGENCE_DIAGNOSTIC'
    except BaseException as e:
        result.update(status='STOPPED_INTERLEAVED_CONVERGENCE_DIAGNOSTIC', error_type=type(e).__name__, error=str(e))
        raise
    finally:
        result.update(wall_seconds=time.time()-start, assigned_chips=0, optimizer_updates=0, heldout_reads=0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, scope=p['scope'])
        write(out / 'report.json', result); write(PUBLIC / 'report.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--action', choices=['prepare', 'run'], required=True)
    a = p.parse_args(); (prepare if a.action == 'prepare' else run)(a.root)
