"""Exact causal readout of the only trained fixed-one budget."""
import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def guard(root, p):
    if time.time() > p['absolute_deadline_unix']:
        raise TimeoutError('Immutable depth-causal deadline')
    mem = {r.split(':')[0]: int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable'] < 512*1024**3 or min(shutil.disk_usage(root).free, shutil.disk_usage('/').free) < 35*1024**3:
        raise RuntimeError('Protected memory/disk reserve')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 6*1024**3:
        raise RuntimeError('Six GiB CPU worker RSS bound')


def indices_for_worker(rank, workers, count=384):
    if not 0 <= rank < workers:
        raise ValueError('Frozen worker index')
    return [list(range(left, min(left+2, count))) for left in range(rank*2, count, workers*2)]


def worker(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from compute_trainium.depth_causal_workspace import DepthCausalWorkspace
    from compute_trainium.model import ModelConfig
    from scripts.analyze_fixed_depth_causal import verify
    p = verify(args.root, args.plan)
    guard(args.root, p)
    start = time.time()
    rank, workers = args.worker, p['exact_readout_workers']
    batches = indices_for_worker(rank, workers)
    torch.set_num_threads(p['exact_readout_threads'])
    torch.set_num_interop_threads(1)
    selection = json.loads((args.root/p['identity']/'select/report.json').read_text())['selection']
    arm = p['arms'][selection['arm']]
    checkpoint = args.root/'runs'/arm['phase']/f"PRIVATE-u{arm['checkpoint_updates'][-1]}.pt"
    audit = json.loads((args.root/p['identity']/('audit-'+selection['arm'])/'report.json').read_text())
    if {r['checkpoint_sha256'] for r in audit['rows']} != {file_sha256(checkpoint)}:
        raise AssertionError('Audited endpoint changed')
    evaluation = args.root/p['evaluation_relative']
    if any(file_sha256(evaluation/f) != h for f, h in p['evaluation_files'].items()):
        raise AssertionError('Original DEV panel changed')
    tokens = np.load(evaluation/'broad-tokens.npy', mmap_mode='r')
    meta = json.loads((evaluation/'PRIVATE-broad-metadata.json').read_text())
    if len(tokens) != 384 or len(meta) != 384:
        raise AssertionError('Complete original panel required')
    model = DepthCausalWorkspace(ModelConfig(**p['model_config'])).double().eval()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(saved['model'])
    del saved
    rows, checks = [], []
    with torch.inference_mode():
        # Every process completes all its reference checks before scoring its
        # assigned targets. The aggregate requires all processes to pass.
        for budget in (1,):
            for i in range(rank, 6, workers):
                guard(args.root, p)
                index = (i//2)*128+i%2
                x = torch.from_numpy(np.array(tokens[index:index+1, :32], dtype=np.int64))
                a = model(x, budget)
                b = model.sequential_reference(x, budget)
                error = float((a.log_softmax(-1)-b.log_softmax(-1)).abs().max())
                checks.append(dict(budget=budget, index=index, maximum_log_probability_error=error))
                if error > 1e-8:
                    raise AssertionError('FP64 cache/full-prefix parity')
                del a, b
        for budget in (1,):
            for domain in range(3):
                guard(args.root, p)
                index = domain*128+rank*2
                x = torch.from_numpy(np.array(tokens[index:index+1, :-1], dtype=np.int64))
                a, b = model(x, budget), model.full_reference(x, budget)
                error = float((a.log_softmax(-1)-b.log_softmax(-1)).abs().max())
                checks.append(dict(budget=budget, index=index, full_source_length=512,
                                   maximum_log_probability_error=error))
                if error > 1e-8:
                    raise AssertionError('FP64 actual-context cache/full-mask parity')
                del a, b
        for budget in (1,):
            for indices in batches:
                guard(args.root, p)
                x = torch.from_numpy(np.array(tokens[indices, :-1], dtype=np.int64))
                y = torch.from_numpy(np.array(tokens[indices, 1:], dtype=np.int64))
                logits = model(x, budget)
                losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none').view_as(y)
                for j, i in enumerate(indices):
                    m = meta[i]
                    rows.append(dict(label=f'b{budget}', index=i, id=m['id'], group=m['group'],
                                     domain=m['domain'], targets=512, loss_sum=float(losses[j].sum())))
                del logits, losses
                write_state(args.output/f'worker{rank}-progress.json', dict(budget=budget,
                    completed_rows=len(rows), wall_seconds=time.time()-start))
    write_state(args.output/f'PRIVATE-worker{rank}-scores.json', rows)
    write_state(args.output/f'worker{rank}-report.json', dict(status='COMPLETE_FIXED_DEPTH_CAUSAL_EXACT_WORKER',
        checks=checks, rows=len(rows), wall_seconds=time.time()-start, checkpoint_sha256=file_sha256(checkpoint),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        assigned_chips=0, optimizer_updates=0, protected_groups_scored=0))


def run(args):
    from scripts.analyze_fixed_depth_causal import verify, summary
    p = verify(args.root, args.plan)
    guard(args.root, p)
    start = time.time()
    args.output.mkdir(exist_ok=False)
    jobs, logs = [], []
    try:
        for rank in range(p['exact_readout_workers']):
            log = (args.output/f'worker{rank}.log').open('w')
            logs.append(log)
            jobs.append(subprocess.Popen([sys.executable, '-m', 'scripts.readout_fixed_depth_causal',
                '--root', str(args.root), '--plan', str(args.plan), '--output', str(args.output), '--worker', str(rank)],
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'USE_TORCH_XLA': '0', 'PYTHONDONTWRITEBYTECODE': '1'}))
        while any(j.poll() is None for j in jobs):
            guard(args.root, p)
            if time.time()-start > p['exact_readout_seconds_limit']:
                raise TimeoutError('Frozen exact CPU readout cap')
            if any(j.poll() not in (None, 0) for j in jobs):
                raise RuntimeError('Exact readout worker failed')
            time.sleep(5)
        if any(j.returncode for j in jobs):
            raise RuntimeError('Exact readout worker failed')
        files = [args.output/f'PRIVATE-worker{rank}-scores.json' for rank in range(p['exact_readout_workers'])]
        rows = [r for f in files for r in json.loads(f.read_text())]
        receipts = [json.loads((args.output/f'worker{rank}-report.json').read_text()) for rank in range(p['exact_readout_workers'])]
        if len({r['checkpoint_sha256'] for r in receipts}) != 1 or sum(len(r['checks']) for r in receipts) != 18:
            raise AssertionError('Same checkpoint and complete preflight checks required')
        selection = json.loads((args.root/p['identity']/'select/report.json').read_text())['selection']
        write_state(args.output/'report.json', dict(status='COMPLETE_FIXED_DEPTH_CAUSAL_EXACT_READOUT',
            selected_arm=selection['arm'], scores={f'b{b}': summary([r for r in rows if r['label'] == f'b{b}']) for b in (1,)},
            workers=receipts, private_score_sha256={f.name: file_sha256(f) for f in files},
            assigned_chips=0, optimizer_updates=0, protected_groups_scored=0, wall_seconds=time.time()-start))
    finally:
        for j in jobs:
            if j.poll() is None:
                j.terminate()
        for j in jobs:
            try:
                j.wait(timeout=10)
            except subprocess.TimeoutExpired:
                j.kill()
                j.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    run(args) if args.worker is None else worker(args)
