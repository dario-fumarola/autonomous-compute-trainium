"""Separate budget-training penalty from a change in the causal memory graph."""
import argparse
import json
from pathlib import Path
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.analyze_depth_causal import verify, summary
from scripts.analyze_ephemeral_tuning import read_scores, select
from scripts.compare_interleaved_tied import compare


def decisions(comparisons, bracketed, schedule_margin=.02, ordinary_margin=.01):
    return {label: bool(bracketed and comparisons[label]['comparison']['gain'] >= margin
                       and comparisons[label]['comparison']['conditional_95_interval'][0] > 0)
            for label, margin in [('fixed_over_mixed', schedule_margin), ('fixed_over_strong_ordinary', ordinary_margin)]}


def selection(root, p):
    folder = root/'runs'/(p['identity']+'-evaluation')
    scores = [dict(arm=name, lr=arm['lr'], **summary(read_scores(folder, name+'-b1')))
              for name, arm in p['arms'].items()]
    return dict(status='COMPLETE_FIXED_DEPTH_CAUSAL_LR_SELECTION',
                selection=select(scores, p['learning_rate_grid']), all_scores=scores,
                selection_readout='Only the trained one-step exit, followed by independent exact FP64 readout.',
                automatic_promotion=False, protected_groups_scored=0)


def comparison(root, p):
    out = root/p['identity']
    selected = json.loads((out/'select/report.json').read_text())['selection']
    receipt = json.loads((out/'readout/report.json').read_text())
    if receipt['status'] != 'COMPLETE_FIXED_DEPTH_CAUSAL_EXACT_READOUT' or receipt['selected_arm'] != selected['arm']:
        raise AssertionError('Completed exact selected endpoint required')
    for f,h in receipt['private_score_sha256'].items():
        if file_sha256(out/'readout'/f) != h:
            raise AssertionError('Exact scores changed')
    fixed = [r for f in sorted((out/'readout').glob('PRIVATE-worker*-scores.json')) for r in json.loads(f.read_text())]
    if any(r['label'] != 'b1' for r in fixed):
        raise AssertionError('Only the trained exit may be scored')
    mixed = [r for f in sorted((root/'depth-causal-tuning-v1/readout').glob('PRIVATE-worker*-scores.json'))
             for r in json.loads(f.read_text()) if r['label'] == 'b1']
    persistent = [r for f in sorted((root/'interleaved-mixed-sequential-dev-v1').glob('PRIVATE-worker*-scores.json'))
                  for r in json.loads(f.read_text())]
    tied = json.loads((root/'interleaved-tied-ordinary-tuning-v1/analysis/report.json').read_text())['selection']
    ordinary = read_scores(root/'runs/interleaved-tied-ordinary-tuning-v1-evaluation', tied['arm']+'-main')
    kw = dict(seed=p['bootstrap_seed'], replicates=p['bootstrap_replicates'])
    comparisons = dict(fixed_over_mixed=compare(fixed,mixed,**kw),
                       fixed_over_strong_ordinary=compare(fixed,ordinary,**kw),
                       fixed_over_persistent_fixed=compare(fixed,persistent,**kw))
    gates = decisions(comparisons, selected['bracketed'], p['budget_training_gain_nats'], p['practical_gain_nats'])
    arm = p['arms'][selected['arm']]
    fit = json.loads((root/'runs'/arm['phase']/'report.json').read_text())
    return dict(status='COMPLETE_FIXED_DEPTH_CAUSAL_ANALYSIS', selection=selected,
        fixed_one_score=summary(fixed), mixed_one_score=summary(mixed),
        comparisons=comparisons, decisions=gates, both_development_gates_pass=all(gates.values()),
        resources=dict(parameters=fit['parameters'], real_targets=fit['real_targets'],
                       nominal_training_matrix_flops=fit['nominal_training_matrix_flops'],
                       nominal_work_ratio_to_strong_ordinary=fit['nominal_training_matrix_flops']/43_812_370_828_492_800,
                       native_fit_wall_seconds=fit['wall_seconds']),
        trained_seeds=1, protected_groups_scored=0, terminal_alpha_spent=0, automatic_promotion=False,
        previous_multi_budget_gates_remain_failed=True,
        uncertainty_scope='Conditional source-group intervals on reused development sources and independently LR-selected single-seed endpoints. Not training-seed uncertainty or fresh confirmation.',
        scope=p['scope'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--action', choices=['select','compare'], required=True)
    args = parser.parse_args()
    start = time.time()
    p = verify(args.root, args.plan)
    result = (selection if args.action == 'select' else comparison)(args.root,p)
    result['wall_seconds'] = time.time()-start
    if time.time() > p['absolute_deadline_unix']:
        raise TimeoutError('Immutable experiment deadline')
    args.output.mkdir(exist_ok=False)
    write_state(args.output/'report.json',result)
