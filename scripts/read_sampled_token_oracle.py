"""CPU replay and hindsight selection over the existing sampled-depth outputs."""
import argparse
import concurrent.futures as cf
import itertools
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

from compute_trainium.document_depth_oracle import hindsight_point
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY = 'sampled-token-depth-oracle-v1'
PUBLIC = Path('results/future_span_depth_oracle_20261025/token')
SEEDS = [5231, 7283]
DEPTHS = [1, 2, 3]
WEIGHTS = {'web': .6, 'math': .2, 'code': .2}


def read(path):
    return json.loads(Path(path).read_text())


def software_fixture():
    # Numerical fixture only. These values never enter any research corpus.
    x = np.array([[1., 3., 2.], [2., 1., 2.], [3., 2., 1.]])
    w = np.array([.6, .2, .2])
    result = hindsight_point(x*w, np.ones_like(x), 0.)
    exhaustive = min(sum(w[j]*x[i, j] for j, i in enumerate(c))
                     for c in itertools.product(range(3), repeat=3))
    assert abs(result['weighted_nll']-exhaustive) < 1e-14
    assert result['choice'].tolist() == [0, 1, 2]
    assert np.argmin(np.ones((3, 2)), axis=0).tolist() == [0, 0]


def register(root):
    out = root/IDENTITY
    if out.exists() or PUBLIC.exists():
        raise FileExistsError('Preserve every registered diagnostic and failed attempt')
    main = root/'future-span-main-v2'
    doc = root/'future-span-depth-oracle-v2'
    work = root/'future-span-work-control-v1'
    ordinary = root/'lean-baseline-main-ordinary8-v1'
    mp, dr, op = read(main/'plan.json'), read(doc/'report.json'), read(ordinary/'report.json')
    assert read(main/'report.json')['status'] == 'COMPLETE_BOUNDED_FUTURE_SPAN_OBJECTIVE_PILOT'
    assert dr['status'] == 'COMPLETE_RETROSPECTIVE_FULL_GRADIENT_DOCUMENT_BOUND'
    doc_plan = read(doc/'plan.json')
    assert doc_plan['main_report_sha256'] == file_sha256(main/'report.json')
    assert doc_plan['main_plan_sha256'] == file_sha256(main/'plan.json')
    assert dr['plan_sha256'] == file_sha256(doc/'plan.json')
    assert read(work/'report.json')['status'] == 'COMPLETE_WITHIN_FAMILY_WORK_MATCHED_CONTROLS'
    assert op['status'] == 'COMPLETE_QUALIFIED_LEAN_BASELINE_MAIN_CONTROL'
    assert dr['parent_work_report_sha256'] == file_sha256(work/'report.json')
    assert mp['evaluation']['scope'] == 'consumed_development_only'
    assert mp['evaluation']['corpus'] == 'variable-depth-fresh-v1/dev.json'
    jobs = []
    for seed in SEEDS:
        case = f'recurrent-ntp-s{seed}-lr32000'
        checkpoint = root/'runs'/f'future-span-main-v2-{case}-finish/checkpoint-u1920.pt'
        for depth in DEPTHS:
            score = main/'scores'/f'{case}-cooled1b-d{depth}'
            row = read(score/'report.json')
            assert row['seed'] == seed and row['depth'] == depth
            assert row['training_real_targets'] == 1004887590
            assert row['checkpoint_sha256'] == file_sha256(checkpoint)
            jobs.append(dict(seed=seed, depth=depth, case=case,
                checkpoint=str(checkpoint.relative_to(root)), checkpoint_sha256=row['checkpoint_sha256'],
                original_score=str(score.relative_to(root)), original_score_sha256=file_sha256(score/'report.json'),
                original_documents_sha256=file_sha256(score/'document-scores.json'),
                prior_score_seconds=row['wall_seconds']))
    sources = {p: file_sha256(p) for p in [__file__, 'scripts/evaluate_fixed_two_curve_cpu.py',
        'compute_trainium/document_depth_oracle.py', 'compute_trainium/future_span_evaluation.py']}
    # Keep the source path relative in public receipts.
    sources = {str(Path(k).relative_to(Path.cwd()) if Path(k).is_absolute() else Path(k)): v for k, v in sources.items()}
    plan = dict(identity=IDENTITY, registered_unix=time.time(), source_sha256=sources,
        main_plan_sha256=file_sha256(main/'plan.json'), main_report_sha256=file_sha256(main/'report.json'),
        document_oracle_sha256=file_sha256(doc/'report.json'), work_plan_sha256=file_sha256(work/'plan.json'),
        work_report_sha256=file_sha256(work/'report.json'), ordinary_report_sha256=file_sha256(ordinary/'report.json'),
        jobs=jobs, seeds=SEEDS, depths=DEPTHS, weights=WEIGHTS, max_cpu_workers=2,
        per_replay_seconds_limit=900, total_seconds_limit=2400,
        measured_prior_parallel_score_seconds=max(sum(j['prior_score_seconds'] for j in jobs if j['seed'] == s) for s in SEEDS),
        aggregate_reproduction_absolute_nll_tolerance=1e-8,
        per_document_reproduction_mean_nll_tolerance=1e-7,
        already_saved_sampled_depth_token_losses=False, model_updates=0, assigned_chips=0,
        test_accessed=False, terminal_alpha_spent=False, new_training_or_router_admitted=False,
        scope='Retrospective consumed-DEV diagnostic chosen after conventional-control results. Reuse the unchanged CPU position-loss evaluator and require every original depth/document score to reproduce before calculating minima. Select each real target\'s smallest loss among recorded uniform-depth1/2/3 outputs, with targets and selection cost free. This bounds only postselection among these outputs, not changed token-routing trajectories, other checkpoints, training or architectures. Original matched-work controls use the same training seeds and budget. Ordinary8 is a descriptive higher-training-budget reference, with seed7283 lacking a same-seed ordinary8 endpoint; no paired training-seed interval or matched-budget claim against it. Private loss/index arrays stay private. No fresh reserve,TEST,alpha or automatic model experiment.')
    if plan['measured_prior_parallel_score_seconds']*2+180 > plan['total_seconds_limit']:
        raise RuntimeError('Measured complete CPU replay cannot fit the bounded cap')
    out.mkdir(); PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', plan); write_state(PUBLIC/'registration.json', plan)
    return plan


def analyze(root, plan):
    out = root/IDENTITY
    doc = read(root/'future-span-depth-oracle-v2/report.json')
    ordinary = read(root/'lean-baseline-main-ordinary8-v1/report.json')
    ordinary_mean = float(np.mean([v['metrics']['weighted_nll'] for v in ordinary['scores'].values()]))
    ordinary_work = {s: v['training_executed_matmuls'] for s, v in ordinary['scores'].items()}
    rows = []; inputs = []; canonical = None
    for seed in SEEDS:
        matrices = []; pairing = None; reports = []
        for depth in DEPTHS:
            job = next(j for j in plan['jobs'] if j['seed'] == seed and j['depth'] == depth)
            folder = out/f's{seed}-d{depth}'; report = read(folder/'report.json')
            old = read(root/job['original_score']/'report.json')
            assert file_sha256(root/job['original_score']/'report.json') == job['original_score_sha256']
            assert file_sha256(root/job['original_score']/'document-scores.json') == job['original_documents_sha256']
            index = read(folder/'token-loss-index.json')['records']
            array = np.load(folder/'token-losses.npy', mmap_mode='r')
            assert file_sha256(folder/'token-losses.npy') == report['token_losses_sha256']
            assert file_sha256(folder/'token-loss-index.json') == report['token_loss_index_sha256']
            assert report['checkpoint_sha256'] == job['checkpoint_sha256']
            assert report['seed'] == seed and report['depth'] == depth and report['cohort'] == 'curve'
            assert report['estimated_training_dense_matrix_ops'] == old['estimated_training_dense_matrix_ops']
            assert report['source_corpus_sha256'] == old['source_corpus_sha256']
            assert report['model_unchanged'] and report['full_state_readback_exact']
            assert abs(report['metrics']['weighted_nll']-old['metrics']['weighted_nll']) <= plan['aggregate_reproduction_absolute_nll_tolerance']
            original = sorted(read(root/job['original_score']/'document-scores.json')['records'], key=lambda r: r['id'])
            assert len(index) == len(original) == 603 and len(array) == 754750
            assert np.isfinite(array).all() and (array >= 0).all()
            expected_offset = 0
            for frame, previous in zip(index, original):
                assert (frame['id'], frame['group'], frame['domain'], frame['length']) == (previous['id'], previous['group'], previous['domain'], previous['real_targets'])
                assert frame['offset'] == expected_offset; expected_offset += frame['length']
                value = float(array[frame['offset']:expected_offset].sum())
                assert abs(value-previous['nll_sum'])/frame['length'] <= plan['per_document_reproduction_mean_nll_tolerance']
            assert expected_offset == len(array)
            if pairing is not None: assert index == pairing
            pairing = index; matrices.append(array); reports.append(report)
            inputs.append(dict(seed=seed, depth=depth, replay_report_sha256=file_sha256(folder/'report.json'),
                token_losses_sha256=report['token_losses_sha256'], index_sha256=report['token_loss_index_sha256']))
        if canonical is not None: assert pairing == canonical
        canonical = pairing
        matrix = np.stack(matrices); weights = np.zeros(matrix.shape[1]); totals = {}
        for domain, weight in WEIGHTS.items():
            totals[domain] = sum(f['length'] for f in pairing if f['domain'] == domain)
            for f in pairing:
                if f['domain'] == domain: weights[f['offset']:f['offset']+f['length']] = weight/totals[domain]
        assert np.all(weights > 0) and abs(weights.sum()-1) < 1e-12
        selected = hindsight_point(matrix*weights, np.ones_like(matrix), 0.)
        choice = selected['choice']; chosen = matrix[choice, np.arange(matrix.shape[1])]
        token_nll = float(np.dot(chosen, weights))
        assert abs(token_nll-selected['weighted_nll']) < 1e-12
        constant = {str(d): float(np.dot(matrix[i], weights)) for i, d in enumerate(DEPTHS)}
        for i, d in enumerate(DEPTHS): assert abs(constant[str(d)]-reports[i]['metrics']['weighted_nll']) < 1e-10
        document_nll = 0.
        domains = {d: 0. for d in WEIGHTS}
        for f in pairing:
            start, stop = f['offset'], f['offset']+f['length']
            document_nll += WEIGHTS[f['domain']]*float(matrix[:, start:stop].sum(axis=1).min())/totals[f['domain']]
            domains[f['domain']] += float(chosen[start:stop].sum())/totals[f['domain']]
        old_doc = next(r for r in doc['records'] if r['seed'] == seed)
        assert abs(document_nll-old_doc['points'][0]['weighted_nll']) < 1e-8
        assert token_nll <= document_nll+1e-12 and token_nll <= min(constant.values())+1e-12
        assert abs(sum(WEIGHTS[d]*v for d, v in domains.items())-token_nll) < 1e-12
        matched = [dict(cell=c['cell'], control_nll=c['nll'], free_token_oracle_gain=c['nll']-token_nll) for c in old_doc['controls']]
        work = reports[0]['estimated_training_dense_matrix_ops']
        rows.append(dict(seed=seed, token_oracle_nll=token_nll, document_oracle_nll=document_nll,
            gain_over_document_selection=document_nll-token_nll, constant_depth_nll=constant,
            chosen_token_counts={str(d): int((choice == i).sum()) for i, d in enumerate(DEPTHS)},
            domain_oracle_nll=domains, matched_original_controls=matched, training_matrix_work=work,
            higher_budget_ordinary8_mean_reference_nll=ordinary_mean,
            descriptive_reference_gain=ordinary_mean-token_nll,
            ordinary8_training_work_ratio={s: w/work for s, w in ordinary_work.items()}))
    return dict(status='COMPLETE_RETROSPECTIVE_TOKEN_OUTPUT_SELECTION_BOUND', records=rows,
        mean_token_oracle_nll=float(np.mean([r['token_oracle_nll'] for r in rows])),
        descriptive_gain_over_higher_budget_ordinary8=ordinary_mean-float(np.mean([r['token_oracle_nll'] for r in rows])),
        ordinary8_mean_reference_nll=ordinary_mean, input_hashes=inputs,
        new_model_window_forwards=6*1050,
        all_original_document_depth_scores_reproduced=True, target_selection_cost_charged=False,
        deployable_routing_cost_established=False, all_adaptive_computation_bounded=False,
        new_training_or_router_admitted=False, terminal_success=False, scope=plan['scope'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    software_fixture(); out = args.root/IDENTITY
    plan = read(out/'plan.json') if (out/'plan.json').exists() else register(args.root)
    if args.prepare_only: return
    if (out/'started.json').exists(): raise FileExistsError('Preserve stopped analyses; explicit recovery required')
    started = time.time(); abort = threading.Event(); phases = []; lock = threading.Lock()
    plan_sha = file_sha256(out/'plan.json')
    write_state(out/'started.json', dict(unix=started, plan_sha256=plan_sha))
    result = dict(status='STOPPED_TOKEN_OUTPUT_SELECTION_BOUND')
    def interrupted(number, frame):
        abort.set(); raise KeyboardInterrupt('CPU token replay interrupted: '+str(number))
    handlers = {s: signal.signal(s, interrupted) for s in [signal.SIGINT, signal.SIGTERM]}
    def guard():
        if abort.is_set() or time.time()-started > plan['total_seconds_limit']: raise TimeoutError('Bounded CPU replay stopped')
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available < 544*1024**3 or min(shutil.disk_usage(args.root).free, shutil.disk_usage('/').free) < 35*1024**3: raise RuntimeError('Protected CPU replay resources')
        if file_sha256(out/'plan.json') != plan_sha: raise AssertionError('Frozen diagnostic plan changed')
        if any(file_sha256(f) != h for f, h in plan['source_sha256'].items()): raise AssertionError('Frozen diagnostic source changed')
        for parent, key in [('future-span-main-v2/plan.json', 'main_plan_sha256'),
                            ('future-span-main-v2/report.json', 'main_report_sha256'),
                            ('future-span-depth-oracle-v2/report.json', 'document_oracle_sha256'),
                            ('future-span-work-control-v1/plan.json', 'work_plan_sha256'),
                            ('future-span-work-control-v1/report.json', 'work_report_sha256'),
                            ('lean-baseline-main-ordinary8-v1/report.json', 'ordinary_report_sha256')]:
            if file_sha256(args.root/parent) != plan[key]: raise AssertionError('Frozen parent result changed')
    def seed_replay(seed):
        for job in [j for j in plan['jobs'] if j['seed'] == seed]:
            guard(); folder = out/f's{seed}-d{job["depth"]}'; begin = time.time()
            cmd = [sys.executable, '-B', '-u', '-m', 'scripts.evaluate_fixed_two_curve_cpu', '--root', str(args.root),
                '--plan', str(args.root/'future-span-main-v2/plan.json'), '--plan-sha', plan['main_plan_sha256'],
                '--checkpoint', str(args.root/job['checkpoint']), '--checkpoint-sha', job['checkpoint_sha256'],
                '--case', job['case'], '--cohort', 'curve', '--depth', str(job['depth']), '--output', str(folder),
                '--upstream', str(args.root/'upstream-parcae-69284c1'), '--seconds-limit', str(plan['per_replay_seconds_limit'])]
            process = None
            try:
                with (out/f's{seed}-d{job["depth"]}.log').open('x') as log:
                    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    while process.poll() is None:
                        guard(); time.sleep(1)
                    if process.returncode: raise RuntimeError('CPU replay failed: '+folder.name)
            except BaseException as error:
                with lock:
                    if not (out/'first-phase-failure.json').exists():
                        write_state(out/'first-phase-failure.json', dict(seed=seed, depth=job['depth'],
                            error=str(error), error_kind=type(error).__name__, traceback=traceback.format_exc()))
                abort.set(); raise
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
                with lock: phases.append(dict(seed=seed, depth=job['depth'], wall_seconds=time.time()-begin,
                    exit_code=None if process is None else process.returncode, assigned_chips=0))
    try:
        with cf.ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(seed_replay, s) for s in SEEDS]
            for f in futures: f.add_done_callback(lambda f: abort.set() if f.exception() else None)
            for f in futures: f.result()
        guard(); result = analyze(args.root, plan)
    except BaseException as error:
        abort.set(); result.update(error=str(error), error_kind=type(error).__name__, traceback=traceback.format_exc())
        if (out/'first-phase-failure.json').exists(): result['first_phase_failure'] = read(out/'first-phase-failure.json')
        write_state(out/'initiating-failure.json', result); raise
    finally:
        for s, handler in handlers.items(): signal.signal(s, handler)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(plan_sha256=file_sha256(out/'plan.json'), phases=phases, wall_seconds=time.time()-started,
            child_cpu_seconds=usage.ru_utime+usage.ru_stime, assigned_chips=0, optimizer_updates=0,
            test_accessed=False, terminal_alpha_spent=False,
            cost_scope='CPU-only replay overlaps admitted native training;do not double-add instance opportunity or describe CPU forwards as free.')
        write_state(out/'report.json', result); write_state(PUBLIC/'report.json', result)
        print(json.dumps({k: result[k] for k in ['status', 'wall_seconds', 'assigned_chips']}))


if __name__ == '__main__':
    main()
