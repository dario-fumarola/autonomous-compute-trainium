"""TRAIN-only numerical consistency for both actual generated-token budgets."""
import argparse
import hashlib
import json
import math
import resource
import shutil
import time
from pathlib import Path

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def consistency_decisions(rows, limits):
    domains = {}
    for budget in (1, 2):
        primary = 3 if budget == 1 else 5
        for domain in ('web', 'math', 'code'):
            main = [r for r in rows if r['budget'] == budget and r['domain'] == domain and r['round'] == primary]
            tail = [r for r in rows if r['budget'] == budget and r['domain'] == domain and r['round'] == 32]
            if len(main) != 2 or len(tail) != 2:
                raise AssertionError('Two complete TRAIN windows per domain/budget')
            shift = sum(r['nll']-r['exact_nll'] for r in main)/2
            kl = sum(r['kl_to_sequential'] for r in main)/2
            stability = sum(r['k16_k32_kl'] for r in tail)/2
            if not all(math.isfinite(v) for v in (shift, kl, stability)):
                raise AssertionError('Finite numerical-consistency results required')
            domains[f'{budget}-{domain}'] = dict(nll_shift=shift, kl_to_sequential=kl, k16_k32_kl=stability,
                passed=abs(shift) <= limits['maximum_absolute_domain_nll_shift'] and kl <= limits['maximum_domain_kl']
                       and stability <= limits['maximum_domain_k16_k32_kl'])
    return domains


def main(args):
    import torch
    from torch.nn import functional as F
    from compute_trainium.budgeted_interleaved_factory import make_model
    from compute_trainium.budgeted_interleaved_cache import cached_budget
    from compute_trainium.indexed_natural_sources import IndexedNaturalSources
    p = json.loads(args.plan.read_text())
    if file_sha256(args.plan) != args.plan_sha:
        raise AssertionError('Frozen scientific plan changed')
    if any(file_sha256(f) != h for f, h in p['source_sha256'].items()):
        raise AssertionError('Frozen consistency source changed')
    arm = p['arms'][args.arm]
    audit = json.loads((args.root/p['identity']/('audit-'+args.arm)/'report.json').read_text())
    if audit['status'] != 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':
        raise AssertionError('Every trained graph requires mature audit first')
    checkpoint = args.root/'runs'/arm['phase']/f"PRIVATE-u{arm['checkpoint_updates'][-1]}.pt"
    if {r['checkpoint_sha256'] for r in audit['rows']} != {file_sha256(checkpoint)}:
        raise AssertionError('Audited endpoint changed')
    start = time.time()
    args.output.mkdir()
    torch.set_num_threads(16)
    model = make_model(p).eval()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(saved['model'])
    del saved
    data = IndexedNaturalSources(args.root/p['data_relative'])
    rows, checks = [], []

    def guard():
        if time.time() > p['absolute_deadline_unix'] or time.time()-start > 1200:
            raise TimeoutError('Registered TRAIN consistency bound')
        mem = {x.split(':')[0]: int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
        if mem['MemAvailable'] < 512*1024**3 or min(shutil.disk_usage(args.root).free, shutil.disk_usage('/').free) < 35*1024**3:
            raise RuntimeError('Protected resource reserve')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 16*1024**3:
            raise RuntimeError('16GiB consistency RSS bound')

    def nll(lp, y):
        return -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)

    with torch.inference_mode():
        for budget in (1, 2):
            primary = 3 if budget == 1 else 5
            for view in p['consistency_windows']:
                guard()
                raw, _ = data.window(view['index'])
                if hashlib.sha256(raw.astype('<i8').tobytes()).hexdigest() != view['tokens_sha256']:
                    raise AssertionError('Frozen TRAIN window changed')
                x, y, mask = data.batch([view['index']])
                if not bool(mask.all()):
                    raise AssertionError('All-real-target TRAIN window required')
                a = cached_budget(model, x[:, :32], budget).double().log_softmax(-1)
                b = model.sequential_reference(x[:, :32], budget).double().log_softmax(-1)
                err = float((nll(a, y[:, :32])-nll(b, y[:, :32])).abs().max())
                if err > .0002:
                    raise AssertionError('Actual-prefix cached/full reference parity')
                checks.append(dict(budget=budget, domain=view['domain'], maximum_target_loss_error=err))
                del a, b
                exact = cached_budget(model, x, budget).double().log_softmax(-1)
                exact_nll = float(nll(exact, y).mean())
                source = model.embeddings(x)
                initial = model.body_at(source, torch.arange(x.shape[1]))
                states = tuple(initial for _ in range(budget))
                at16 = None
                for k in range(33):
                    guard()
                    if k in p['consistency_rounds']:
                        output = model.expanded_body(source, states)[budget]
                        lp = F.linear(output, model.prediction_weight).double().log_softmax(-1)
                        row = dict(budget=budget, domain=view['domain'], round=k,
                                   nll=float(nll(lp, y).mean()), exact_nll=exact_nll,
                                   kl_to_sequential=float((exact.exp()*(exact-lp)).sum(-1).mean()))
                        if k == 16: at16 = lp.clone()
                        if k == 32:
                            row['k16_k32_kl'] = float((at16.exp()*(at16-lp)).sum(-1).mean())
                        if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
                            raise AssertionError('Nonfinite consistency measurement')
                        rows.append(row)
                        del lp, output
                    if k < 32:
                        states = model.expanded_body(source, states)[:budget]
                del exact, source, initial, states, at16
                write_state(args.output/'progress.json', dict(budget=budget, completed_windows=len(checks), seconds=time.time()-start))
    guard()
    domains = consistency_decisions(rows, p['consistency_guard'])
    passed = all(r['passed'] for r in domains.values())
    report = dict(status='PASS_SUPPORTED_BUDGET_TRAIN_CONSISTENCY' if passed else 'FAIL_SUPPORTED_BUDGET_TRAIN_CONSISTENCY',
                  arm=args.arm, rows=rows, domains=domains, cache_checks=checks,
                  checkpoint_sha256=file_sha256(checkpoint), assigned_chips=0, optimizer_updates=0,
                  heldout_reads=0, wall_seconds=time.time()-start)
    write_state(args.output/'report.json', report)
    if not passed:
        raise AssertionError('Registered sequential-consistency gate failed; no quality readout')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
