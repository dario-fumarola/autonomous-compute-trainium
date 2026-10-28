"""Calibration-gated, zero-update prediction-error workspace diagnostic."""
import argparse
import json
import os
import time
from pathlib import Path

from compute_trainium.training import file_sha256
from scripts.residual_memory_attribution import guard
from scripts.residual_independent_memory_training import load

ID = 'prefix-error-headroom-v2'
PUBLIC = Path('results/prefix_error_headroom_20261028/recovery-v2')
POLICY = Path('docs/PREFIX_ERROR_HEADROOM_20261028.md')
METHODS = ['error', 'uniform', 'label', 'prediction']
RATES = [0., .25, 1., 4., 16.]


def write(out, name, value):
    for folder in [out, PUBLIC]:
        (folder / name).write_text(json.dumps(value, indent=2) + '\n')


def verify(root, plan):
    for name, sha in plan['source_sha256'].items():
        if file_sha256(name) != sha:
            raise AssertionError('Frozen implementation changed')
    for name, sha in plan['retained_feature_sha256'].items():
        if file_sha256(root / name) != sha:
            raise AssertionError('Frozen features changed')
    for ref in plan['bindings'].values():
        if file_sha256(root / ref['relative']) != ref['sha256']:
            raise AssertionError('Frozen input changed')


def prepare(root):
    from compute_trainium.execution_isolation import managed_running
    if managed_running() or os.environ.get('USE_TORCH_XLA') != '0':
        raise RuntimeError('Closed Native owner and CPU mode required')
    start = time.time(); guard(root, start + 1200)
    out = root / ID; out.mkdir(mode=0o700); PUBLIC.mkdir()
    old = json.loads((root / 'causal-associative-workspace-v1/plan.json').read_text())
    bindings = dict(c0=dict(relative=old['checkpoint_relative'], sha256=old['checkpoint_sha256']),
                    source_map=old['source_map'],
                    raw_cache=dict(relative='causal-associative-workspace-v1/PRIVATE-raw-cache.pt',
                                   sha256=file_sha256(root / 'causal-associative-workspace-v1/PRIVATE-raw-cache.pt')),
                    prior_one=old['bindings']['prior_one'],
                    reserve=dict(relative='residual-memory-attribution-v1/PRIVATE-reserved-sources.json',
                                 sha256=old['reserve_receipt']['private_manifest_sha256']))
    source = [POLICY, Path(__file__), Path('compute_trainium/prefix_error_workspace.py'),
              Path('compute_trainium/causal_associative_workspace.py'),
              Path('compute_trainium/stable_target_loss.py'),
              Path('compute_trainium/residual_matrix_work.py'),
              Path('compute_trainium/residual_screen_analysis.py'),
              Path('compute_trainium/dependency_analysis.py'),
              Path('compute_trainium/training.py'),
              Path('scripts/residual_independent_memory_training.py'),
              Path('scripts/residual_memory_attribution.py'),
              Path('tests/test_prefix_error_workspace.py')]
    source = [p.relative_to(Path.cwd()) if p.is_absolute() else p for p in source]
    plan = dict(identity=ID, registered_unix=start, absolute_deadline_unix=start+1200,
                scoring_seconds_limit=900, methods=METHODS, rates=RATES,
                bindings=bindings, feature_segments=old['feature_segments'],
                retained_feature_sha256=old['retained_feature_sha256'],
                evaluation_groups=old['evaluation_groups'],
                source_sha256={str(p): file_sha256(p) for p in source},
                scope=POLICY.read_text(), assigned_chips=0, optimizer_updates=0,
                retained_failure=dict(relative='prefix-error-headroom-v1/report.json',
                     sha256=file_sha256(root/'prefix-error-headroom-v1/report.json')))
    verify(root, plan)
    write(out, 'plan.json', plan)
    write(out, 'admission.json', dict(status='PASS_PREFIX_ERROR_REGISTRATION',
          calibration_views=797, conditional_tune_views=448, reserved_views=0,
          wall_seconds=time.time()-start))


def qualify(features, outcomes, domains, head):
    import torch
    from compute_trainium.prefix_error_workspace import (
        read_weights, read_correction, sequential_bias_reference,
    )
    from compute_trainium.stable_target_loss import stable_target_log_probs
    rows = {}; q = torch.arange(127, 159)
    for domain in range(3):
        i = int((domains == domain).nonzero()[0])
        h = features[i:i+1]; y = outcomes[i:i+1]
        with torch.no_grad():
            logits = h @ head.T; weights = read_weights(h, q)
            actual = read_correction(logits, y, weights)
        reference = sequential_bias_reference(logits.double(), y, weights.double(), q)
        relative = float((actual.double()-reference).norm()/reference.norm().clamp_min(1e-30))
        a = stable_target_log_probs(logits[:, q] + 16*actual, y[:, q])
        b = stable_target_log_probs(logits[:, q].double() + 16*reference, y[:, q])
        error = float((a.double()-b).abs().max())
        if not __import__('math').isfinite(relative) or relative > .02 or error > .0002:
            raise AssertionError('Actual-shape literal bias-gradient gate failed')
        rows[str(domain)] = dict(correction_relative_l2=relative, max_target_logp_error=error)
    return dict(status='PASS_LITERAL_GRADIENT_AUDIT', domains=rows)


def score_view(h, y, head, rates_by_method, *, rotation_rate=None):
    import torch
    from compute_trainium.prefix_error_workspace import read_weights, read_correction, rotated_error
    from compute_trainium.stable_target_loss import stable_target_log_probs
    q = torch.arange(127, 159)
    logits = h @ head.T; target = y[:, q]; raw = stable_target_log_probs(logits[:, q], target)
    weights = read_weights(h, q); uniform = read_weights(h, q, temperature=0.)
    rows = {}
    for method, rates in rates_by_method.items():
        delta = read_correction(logits, y, uniform if method == 'uniform' else weights,
                                mode='error' if method == 'uniform' else method)
        rows[method] = [float(-raw.mean()) if rate == 0 else
                       float(-stable_target_log_probs(logits[:, q].double() + rate*delta, target).mean())
                        for rate in rates]
    if rotation_rate is not None:
        delta = rotated_error(logits, y, weights, q)
        rows['rotated'] = [float(-raw.mean()) if rotation_rate == 0 else
                          float(-stable_target_log_probs(logits[:, q].double() + rotation_rate*delta, target).mean())]
    return rows, float(-raw.mean())


def run(root):
    import torch
    import numpy as np
    from compute_trainium.residual_matrix_work import MatrixWork
    from compute_trainium.residual_screen_analysis import contrast
    from compute_trainium.dependency_analysis import interval
    out = root / ID; plan = json.loads((out/'plan.json').read_text())
    start = time.time(); deadline = min(start+900, plan['absolute_deadline_unix'])
    torch.set_num_threads(16)
    report = dict(status='STOPPED_PREFIX_ERROR_HEADROOM', assigned_chips=0, optimizer_updates=0,
                  tune_scored=False, reserved_sources_scored=False)
    work = MatrixWork()
    try:
        verify(root, plan)
        h, y, d = load(root, plan, 'train', 8192)
        head = torch.load(root/plan['bindings']['c0']['relative'], weights_only=True, mmap=True)['model']['lm_head.weight'].detach()
        version = head._version
        maps = torch.load(root/plan['bindings']['source_map']['relative'], weights_only=True)
        raw_cache = torch.load(root/plan['bindings']['raw_cache']['relative'], weights_only=True)
        with work:
            report['qualification'] = qualify(h, y, d, head)
        write(out, 'numerical-audit.json', report['qualification'])
        indices = maps['calibration'].tolist()
        if len(indices) != 797:
            raise AssertionError('Calibration selection changed')
        sums = np.zeros((4, 5, 3)); counts = np.zeros(3); rows = []; raw_error = 0.
        phase = time.time()
        with torch.no_grad(), work:
            for ni, i in enumerate(indices):
                guard(root, deadline)
                values, raw = score_view(h[i:i+1], y[i:i+1], head, {m:RATES for m in METHODS})
                expected = float(-raw_cache['logp'][i].mean()); raw_error = max(raw_error, abs(raw-expected))
                if raw_error > .0002 or any(abs(v[0]-raw) > 1e-12 for v in values.values()):
                    raise AssertionError('Ordinary zero-rate parity failed')
                domain = int(d[i]); counts[domain] += 1
                for mi, method in enumerate(METHODS): sums[mi, :, domain] += values[method]
                rows.append(dict(index=i, domain=domain, raw=raw, values=values))
                if ni % 16 == 0:
                    (out/'progress.json').write_text(json.dumps(dict(phase='calibration', views=ni+1, wall_seconds=time.time()-start)))
                if ni == 15 and time.time() + (time.time()-phase)/16*(797-16) + 60 > deadline:
                    raise RuntimeError('Measured calibration exceeds registered budget')
        grid = sums/counts
        means = grid @ np.array([.6, .2, .2])
        selected = {m:dict(rate=RATES[int(means[mi].argmin())],
                          nll=float(means[mi].min()), grid_nll=means[mi].tolist(),
                          domain_grid=grid[mi].tolist(), boundary= int(means[mi].argmin()) in [0,4])
                    for mi,m in enumerate(METHODS)}
        raw_mean = float(means[0, 0]); error_nll = selected['error']['nll']
        cal_gain = raw_mean-error_nll
        pass_cal = cal_gain >= .005 and all(selected[m]['nll']-error_nll >= .002 for m in ['label','prediction'])
        report['calibration'] = dict(selected=selected, raw_nll=raw_mean, error_gain=cal_gain,
                    domain_counts=counts.tolist(), gate=pass_cal, max_raw_error=raw_error,
                    wall_seconds=time.time()-phase)
        (out/'PRIVATE-calibration-scores.json').write_text(json.dumps(rows))
        write(out, 'calibration.json', report['calibration'])
        if pass_cal:
            del h, y, d
            h, y, d = load(root, plan, 'tune', 448)
            old = json.loads((root/plan['bindings']['prior_one']['relative']).read_text())
            rows = []; rates = {m:[selected[m]['rate']] for m in METHODS}
            with torch.no_grad(), work:
                for i in range(448):
                    guard(root, deadline)
                    values, raw = score_view(h[i:i+1], y[i:i+1], head, rates,
                                            rotation_rate=selected['error']['rate'])
                    if abs(raw-old[i]['raw_nll']) > .0002:
                        raise AssertionError('Historical TUNE C0 changed')
                    rows.append(dict(index=i, domain=int(d[i]), group=plan['evaluation_groups'][i],
                                     real_targets=32, raw=raw, old_one=old[i]['workspace_nll'],
                                     **{k:v[0] for k,v in values.items()}))
                    (out/'progress.json').write_text(json.dumps(dict(phase='tune',views=i+1)))
            fields = ['raw', 'old_one', *METHODS, 'rotated']
            contrasts = {k:contrast(rows,[r[k]-r['error'] for r in rows]) for k in fields if k!='error'}
            threshold = {'raw':.01, 'label':.005, 'prediction':.005, 'rotated':0.}
            passed = all(contrasts[k]['mean'] >= v and contrasts[k]['conditional_source_group_95_interval'][0]>0 for k,v in threshold.items())
            report['evaluation'] = dict(scores={k:contrast(rows,[r[k] for r in rows]) for k in fields},
                contrasts=contrasts, gate=passed,
                domains={str(di):{k:interval([r for r in rows if r['domain']==di],[r[k] for r in rows if r['domain']==di]) for k in fields} for di in range(3)})
            (out/'PRIVATE-tune-scores.json').write_text(json.dumps(rows))
            report['tune_scored'] = True
        if head._version != version:
            raise AssertionError('Frozen head changed')
        verify(root, plan)
        report.update(status='COMPLETE_PREFIX_ERROR_HEADROOM', parameters_unchanged=True,
                      decision='evaluate_registered_lead' if pass_cal else 'stop_at_calibration_no_fit')
    except BaseException as e:
        report.update(error_type=type(e).__name__, error=str(e))
        raise
    finally:
        report.update(wall_seconds=time.time()-start, matrix_operations=work.operations,
                      plan_sha256=file_sha256(out/'plan.json'))
        write(out, 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',action='store_true'); args = parser.parse_args()
    run(args.root) if args.run else prepare(args.root)
