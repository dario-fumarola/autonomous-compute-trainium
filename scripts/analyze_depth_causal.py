"""Frozen selection and separate practical gates for supported latent budgets."""
import argparse
import json
import time
from pathlib import Path

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.analyze_ephemeral_tuning import read_scores, select
from scripts.analyze_probability_screen import score
from scripts.compare_interleaved_tied import compare, validate_rows


def verify(root, plan):
    p = json.loads(plan.read_text())
    if time.time() > p['absolute_deadline_unix']:
        raise TimeoutError('Registered learning deadline')
    for base, key in [(Path('.'), 'source_sha256'), (root, 'comparison_root_sha256'),
                      (Path('.'), 'comparison_repo_sha256')]:
        if any(file_sha256(base/f) != h for f, h in p[key].items()):
            raise AssertionError('Frozen source or comparison artifact changed')
    for name in p['arms']:
        audit = root/p['identity']/('audit-'+name)/'report.json'
        if json.loads(audit.read_text())['status'] != 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':
            raise AssertionError('Every mature endpoint must pass before scoring')
    return p


def summary(rows):
    validate_rows(rows)
    domains = {d: score([r for r in rows if r['domain'] == d]) for d in ('web', 'math', 'code')}
    return dict(domains=domains, broad_nll=sum(w*domains[d] for d, w in zip(domains, (.6, .2, .2))))


def gates(comparisons, bracketed, practical, noninferiority):
    result = {}
    for label in ('two_over_one', 'two_over_fixed_one', 'two_over_strong_ordinary'):
        c = comparisons[label]['comparison']
        result[label] = bool(bracketed and c['gain'] >= practical and c['conditional_95_interval'][0] > 0)
    # Positive gain means the candidate is better. This is a preregistered
    # development point guard, not a powered non-inferiority conclusion.
    result['zero_exit_point_guard'] = bool(comparisons['zero_over_same_shape_ordinary']['comparison']['gain'] >= -noninferiority)
    return result


def select_endpoint(root, p):
    folder = root/'runs'/(p['identity']+'-evaluation')
    candidates, all_scores = [], []
    for name, arm in p['arms'].items():
        scores = {f'b{b}': summary(read_scores(folder, f'{name}-b{b}')) for b in (0, 1, 2)}
        all_scores.append(dict(arm=name, lr=arm['lr'], budgets=scores))
        candidates.append(dict(arm=name, lr=arm['lr'], **scores['b2']))
    selected = select(candidates, p['learning_rate_grid'])
    return dict(status='COMPLETE_DEPTH_CAUSAL_LR_SELECTION', selection=selected, all_scores=all_scores,
                selection_readout='Two generated tokens in the exact depth-causal graph; independent FP64 cached readout follows.',
                automatic_promotion=False, protected_groups_scored=0)


def compare_endpoint(root, p):
    out = root/p['identity']
    selected = json.loads((out/'select/report.json').read_text())['selection']
    receipt = json.loads((out/'readout/report.json').read_text())
    if receipt['status'] != 'COMPLETE_DEPTH_CAUSAL_EXACT_READOUT' or receipt['selected_arm'] != selected['arm']:
        raise AssertionError('Complete exact selected checkpoint readout required')
    for f, h in receipt['private_score_sha256'].items():
        if file_sha256(out/'readout'/f) != h:
            raise AssertionError('Exact readout changed')
    exact = [r for f in sorted((out/'readout').glob('PRIVATE-worker*-scores.json')) for r in json.loads(f.read_text())]
    rows = {b: [r for r in exact if r['label'] == f'b{b}'] for b in (0, 1, 2)}
    fixed = [r for f in sorted((root/'interleaved-mixed-sequential-dev-v1').glob('PRIVATE-worker*-scores.json'))
             for r in json.loads(f.read_text())]
    tied = json.loads((root/'interleaved-tied-ordinary-tuning-v1/analysis/report.json').read_text())['selection']
    strong = read_scores(root/'runs/interleaved-tied-ordinary-tuning-v1-evaluation', tied['arm']+'-main')
    old = json.loads(Path('results/interleaved_workspace_20261029/tuning-v1/analysis.json').read_text())['selection']
    same = read_scores(root/'runs/interleaved-lr-tuning-v1-evaluation', old['ordinary']['arm']+'-main')
    kw = dict(seed=p['bootstrap_seed'], replicates=p['bootstrap_replicates'])
    comparisons = dict(two_over_one=compare(rows[2], rows[1], **kw),
                       two_over_fixed_one=compare(rows[2], fixed, **kw),
                       two_over_strong_ordinary=compare(rows[2], strong, **kw),
                       zero_over_same_shape_ordinary=compare(rows[0], same, **kw))
    decisions = gates(comparisons, selected['bracketed'], p['practical_gain_nats'], p['zero_exit_noninferiority_margin'])
    resources = {}
    for label, plan, name in [('supported', p, selected['arm']),
        ('fixed_one', json.loads((root/'interleaved-mixed-round-tuning-v1/plan.json').read_text()),
         json.loads((root/'interleaved-mixed-sequential-dev-v1/plan.json').read_text())['selected_arm']),
        ('strong_ordinary', json.loads((root/'interleaved-tied-ordinary-tuning-v1/plan.json').read_text()), tied['arm'])]:
        arm = plan['arms'][name]
        fit = json.loads((root/'runs'/arm['phase']/'report.json').read_text())
        if arm['real_targets'] != p['arms'][selected['arm']]['real_targets']:
            raise AssertionError('Same complete authentic source exposure required')
        resources[label] = dict(parameters=fit['parameters'], real_targets=fit['real_targets'],
            nominal_training_matrix_flops=fit['nominal_training_matrix_flops'],
            native_fit_wall_seconds=fit['wall_seconds'], assigned_fit_chip_hours=16*fit['wall_seconds']/3600)
    return dict(status='COMPLETE_DEPTH_CAUSAL_COMPARISON', selection=selected,
        exact_scores={f'b{b}': summary(rows[b]) for b in rows}, comparisons=comparisons,
        decisions=decisions, all_advancement_gates_pass=all(decisions.values()), resources=resources,
        trained_seeds=1, assigned_chips=0, optimizer_updates=0, protected_groups_scored=0,
        terminal_alpha_spent=0, automatic_promotion=False,
        inference_scope='Exact depth-causal cache, zero/one/two generated states per source token. Earlier stages do not depend on deeper stages. No router, numerical solver, measured serving or training-work efficiency claim.',
        uncertainty_scope='All intervals are conditional on reused development sources and selected single-seed endpoints. The zero-exit guard is a point-estimate development guard. No fresh confirmation.',
        scope=p['scope'])


def main(args):
    start = time.time()
    p = verify(args.root, args.plan)
    result = select_endpoint(args.root, p) if args.action == 'select' else compare_endpoint(args.root, p)
    result['wall_seconds'] = time.time()-start
    if time.time() > p['absolute_deadline_unix']:
        raise TimeoutError('Registered analysis deadline')
    args.output.mkdir(exist_ok=False)
    write_state(args.output/'report.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--action', choices=['select', 'compare'], required=True)
    main(parser.parse_args())
