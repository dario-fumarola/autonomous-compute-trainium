"""Existing exact scorer on a separately frozen elementary-arithmetic panel."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import readout_direct_arithmetic as direct
from scripts.readout_interleaved_bracket_type import checkpoint

ID = 'primitive-arithmetic-readout-v1'
PANEL = 'primitive-arithmetic-panel-v1'
PUBLIC = Path('results/primitive_arithmetic_20261030/readout-v1')
ROLES = ('latent', 'ordinary', 'pause')
BINS = ('add', 'subtract', 'multiply', 'divide')


def read(path):
    return json.loads(path.read_text())


def statistics(rows, replicates=2000, seed=40532):
    import numpy as np
    cells = {role:{r['index']:r for r in rows if r['role'] == role} for role in ROLES}
    reference = cells['latent']; ids = set(reference)
    if not ids or len(rows) != 3*len(ids) or len({r['group'] for r in reference.values()}) != len(ids):
        raise AssertionError('Complete paired unique-source three-cell diagnostic required')
    for cell in cells.values():
        if set(cell) != ids:
            raise AssertionError('Every declared model and case required')
        for i,r in cell.items():
            if any(r[k] != reference[i][k] for k in ('group', 'stratum', 'targets', 'seed')):
                raise AssertionError('Source/target pairing changed')
            if any(not math.isfinite(r[f]) for f in ('answer_nll', 'first_nll', 'exact_prefix_correct')):
                raise AssertionError('Finite scores required')
    indices = {b:sorted(i for i in ids if reference[i]['stratum'] == b) for b in BINS}
    if any(not v for v in indices.values()) or sum(map(len, indices.values())) != len(ids):
        raise AssertionError('Exactly the four frozen operator bins required')
    rng = np.random.default_rng(seed)
    draws = {b:rng.integers(len(v), size=(replicates, len(v))) for b,v in indices.items()}
    def values(role, field):
        return {b:np.array([cells[role][i][field] for i in indices[b]]) for b in BINS}
    def estimate(v):
        samples = sum(v[b][draws[b]].mean(1) for b in BINS)/len(BINS)
        return dict(mean=float(sum(v[b].mean() for b in BINS)/len(BINS)),
                    bins={b:float(v[b].mean()) for b in BINS},
                    conditional_95_interval=np.quantile(samples, [.025,.975]).tolist())
    means = {r:{f:estimate(values(r,f)) for f in ('answer_nll','first_nll','exact_prefix_correct')} for r in ROLES}
    contrasts = {}
    for role in ('ordinary','pause'):
        contrasts[role] = {}
        for f in ('answer_nll','first_nll','exact_prefix_correct'):
            a,b = values('latent',f), values(role,f)
            delta = {k:(a[k]-b[k] if f == 'exact_prefix_correct' else b[k]-a[k]) for k in BINS}
            contrasts[role][f] = estimate(delta)
    return dict(cases=len(ids), bins={b:len(v) for b,v in indices.items()}, means=means, contrasts=contrasts,
        correct_counts={r:sum(int(v['exact_prefix_correct']) for v in cells[r].values()) for r in ROLES},
        training_seeds=1, automatic_promotion=False,
        scope='Exploratory elementary performance, including potentially familiar facts. '
              'Source-disjoint but expression overlap unmeasured; not novel-expression transfer '
              'or training efficiency. Conditional source intervals do not estimate seed uncertainty.')


def prepare(root):
    out = root/ID
    if out.exists() or PUBLIC.exists():
        raise FileExistsError('Preserve every readout attempt')
    receipt = read(root/PANEL/'report.json')
    if (receipt['status'] != 'FROZEN_PRIMITIVE_ARITHMETIC_DEVELOPMENT_PANEL'
            or any(receipt[k] for k in ('source_group_overlap','protected_group_overlap','model_scores'))):
        raise AssertionError('Complete unscored source-isolated panel required')
    points = {}
    for role in ROLES:
        point = checkpoint(root, 'latent-learning-exposure-v1-'+role, role)
        fit = read(root/point['plan_relative'])
        if fit['seed'] != 730271 or fit['arms'][role]['real_targets'] != 1000060368:
            raise AssertionError('Fixed equal-data billion-target models only')
        point.update(seed=730271, role=role); points[role] = point
    files = list(Path('compute_trainium').glob('*.py')) + [
        Path('scripts/readout_primitive_arithmetic.py'), Path('scripts/readout_direct_arithmetic.py'),
        Path('scripts/readout_interleaved_bracket_type.py'), Path('tests/test_primitive_arithmetic_readout.py'),
        Path('docs/PRIMITIVE_ARITHMETIC_DIAGNOSTIC_20261030.md')]
    now = time.time()
    plan = dict(identity=ID, registered_unix=now, absolute_deadline_unix=now+1200,
        threads=8, batch=8, maximum_worker_rss_bytes=6*1024**3, points=points,
        panel_report_sha256=file_sha256(root/PANEL/'report.json'), panel_files=receipt['files'],
        source_sha256={str(f):file_sha256(f) for f in files}, bootstrap_seed=40532, bootstrap_replicates=2000,
        assigned_chips=0, optimizer_updates=0, protected_groups_scored=0, terminal_alpha_spent=0,
        automatic_promotion=False, scope=Path('docs/PRIMITIVE_ARITHMETIC_DIAGNOSTIC_20261030.md').read_text())
    out.mkdir(mode=0o700); PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', plan); write_state(PUBLIC/'plan.json', plan)
    print(json.dumps(dict(status='REGISTERED_PRIMITIVE_READOUT', cases=receipt['cases'], conditions=3)), flush=True)


def worker(root, key):
    direct.ID, direct.PANEL = ID, PANEL
    direct.worker(root, key)


def run(root):
    p = read(root/ID/'plan.json'); direct.ID, direct.PANEL = ID, PANEL
    direct.guard(root,p); started = time.time(); jobs, logs = [], []
    result = dict(status='STOPPED_PRIMITIVE_ARITHMETIC_READOUT')
    try:
        for key in ROLES:
            log = (root/ID/f'{key}.log').open('w'); logs.append(log)
            jobs.append(subprocess.Popen([sys.executable, '-m', 'scripts.readout_primitive_arithmetic',
                '--root', str(root), '--action', 'worker', '--key', key], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'USE_TORCH_XLA':'0', 'PYTHONDONTWRITEBYTECODE':'1'}))
        while any(j.poll() is None for j in jobs):
            direct.guard(root,p)
            if any(j.poll() not in (None,0) for j in jobs):
                raise RuntimeError('Declared readout worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs):
            raise RuntimeError('Declared readout worker failed')
        receipts = {r:read(root/ID/f'{r}-report.json') for r in ROLES}
        if any(v['status'] != 'COMPLETE_DIRECT_ARITHMETIC_WORKER' or len(v['checks']) != 3
               for v in receipts.values()):
            raise AssertionError('All three workers and nine numerical records required')
        rows = [v for r in ROLES for v in read(root/ID/f'PRIVATE-{r}-scores.json')]
        result.update(status='COMPLETE_PRIMITIVE_ARITHMETIC_READOUT', workers=receipts,
                      statistics=statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed']),
                      private_score_sha256={f.name:file_sha256(f)
                          for f in (root/ID).glob('PRIVATE-*-scores.json')})
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        for j in jobs:
            if j.poll() is None:
                j.terminate()
        for j in jobs:
            try: j.wait(timeout=10)
            except subprocess.TimeoutExpired:
                j.kill(); j.wait()
        for log in logs:
            log.close()
        result.update(wall_seconds=time.time()-started, assigned_chips=0, optimizer_updates=0,
            protected_groups_scored=0, terminal_alpha_spent=0, automatic_promotion=False,
            plan_sha256=file_sha256(root/ID/'plan.json'), scope=p['scope'])
        write_state(root/ID/'report.json', result); write_state(PUBLIC/'report.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('scope','workers','statistics')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--action', choices=['prepare','run','worker'], required=True)
    parser.add_argument('--key', choices=ROLES); args = parser.parse_args()
    if args.action == 'worker': worker(args.root,args.key)
    else: (prepare if args.action == 'prepare' else run)(args.root)
