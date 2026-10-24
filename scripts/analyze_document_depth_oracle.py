"""Retrospective DEV bound for whole-document depth choices, never a router."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compute_trainium.document_depth_oracle import weighted_document_losses, hindsight_point
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True)
    started = time.time()
    plan = json.loads(a.registration.read_text())
    for f, expected in plan['source_sha256'].items():
        if file_sha256(f) != expected: raise AssertionError('Registered analysis source changed')
    weights = plan['weights']
    inputs, records = [], []
    canonical_path = Path('results/variable_depth_20261023/dev/summary.json')
    if file_sha256(canonical_path) != plan['canonical_summary_sha256']:
        raise AssertionError('Canonical training-work ledger changed')
    canonical = json.loads(canonical_path.read_text())['rows']
    def load(identity):
        directory = a.root/'runs'/identity
        report_path, scores_path = directory/'report.json', directory/'document-scores.json'
        expected = plan['input_hashes'][identity]
        if (file_sha256(report_path) != expected['report_sha256']
                or file_sha256(scores_path) != expected['document_scores_sha256']):
            raise AssertionError('Registered immutable DEV readout changed')
        report = json.loads(report_path.read_text())
        if report['split'] != 'dev' or report['cohort'] != 'curve' or report['parameters'] != 150436608:
            raise ValueError('Only the registered existing DEV paths are allowed')
        scores = sorted(json.loads(scores_path.read_text()), key=lambda s: s['document'])
        if len({s['document'] for s in scores}) != len(scores): raise ValueError('Duplicate document')
        windows = np.array([math.ceil(s['tokens']/1024) for s in scores])
        if windows.sum() != report['scored_windows'] or sum(s['tokens'] for s in scores) != report['targets']:
            raise AssertionError('Original target/window coverage changed')
        losses = weighted_document_losses(scores, weights)
        if abs(float(losses.sum())-report['metrics']['weighted_nll']) > 1e-10:
            raise AssertionError('Original domain aggregation not reproduced')
        inputs.append({'identity': identity, 'report_sha256': file_sha256(report_path),
                       'document_scores_sha256': file_sha256(scores_path)})
        return report, scores, windows, losses
    for seed in plan['seeds']:
        baseline, normal_docs, windows, ordinary_loss = load(f'vdepth-v1-ordinary-s{seed}-u3840-dev-r1')
        matched, matched_docs, _, matched_loss = load(f'vdepth-v1-ordinary-s{seed}-u6496-dev-r1')
        def identities(scores):
            return [(s['document'], s['group'], s['domain'], s['tokens']) for s in scores]
        pairing = identities(normal_docs)
        if identities(matched_docs) != pairing: raise AssertionError('Ordinary panel pairing differs')
        base_work = float(windows.sum()*baseline['forward_flops_per_window'])
        for family in plan['families']:
            candidate_work = next(r['training_matmul_flops'] for r in canonical
                if r['identity']==f'vdepth-v1-{family}-s{seed}-u3840' and r['depth']==2)
            ordinary_work = next(r['training_matmul_flops'] for r in canonical
                if r['identity']==f'vdepth-v1-ordinary-s{seed}-u6496' and r['depth']==1)
            if abs(ordinary_work/candidate_work-1) > .001:
                raise AssertionError('Registered ordinary training-work match differs')
            scores, costs = [], []
            for depth in plan['depths']:
                report, docs, own_windows, losses = load(f'vdepth-v1-{family}-s{seed}-u3840-dev-r{depth}')
                if identities(docs) != pairing: raise AssertionError('Depth changed document/group/targets')
                scores.append(losses)
                costs.append(own_windows*report['forward_flops_per_window']/base_work)
            score_array, cost_array = np.stack(scores), np.stack(costs)
            points = []
            for penalty in plan['penalties']:
                point = hindsight_point(score_array, cost_array, penalty)
                choices = point.pop('choice')
                point.update(penalty=penalty,
                    document_counts_by_depth={str(d): int((choices == i).sum()) for i, d in enumerate(plan['depths'])},
                    targets_by_depth={str(d): sum(s['tokens'] for j, s in enumerate(normal_docs) if choices[j] == i)
                                      for i, d in enumerate(plan['depths'])})
                points.append(point)
            oracle = points[0]
            records.append({'seed': seed, 'family': family, 'points': points,
                'constant_depth_nll': {str(d): float(scores[i].sum()) for i, d in enumerate(plan['depths'])},
                'ordinary_1B_nll': float(ordinary_loss.sum()), 'ordinary_matched_nll': float(matched_loss.sum()),
                'ordinary_matched_to_candidate_training_work_ratio': ordinary_work/candidate_work,
                'unconstrained_oracle_gain_over_matched_ordinary': float(matched_loss.sum())-oracle['weighted_nll'],
                'documents': len(normal_docs), 'real_targets': sum(s['tokens'] for s in normal_docs)})
    summary = {'status': 'COMPLETE_RETROSPECTIVE_DOCUMENT_ORACLE_DEV_BOUND', 'records': records,
        'input_hashes': inputs, 'wall_seconds': time.time()-started,
        'native_chips_assigned': 0, 'new_model_updates': 0, 'test_accessed': False,
        'new_terminal_attempt': False,
        'scope': plan['scope']}
    write_state(a.output/'summary.json', summary)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = {'fixed': '#bc4b51', 'mixture': '#007f86'}
    for ax, family in zip(axes, plan['families']):
        selected = [r for r in records if r['family'] == family]
        for r in selected:
            points = r['points']
            ax.plot([p['normalized_inference_work'] for p in points], [p['weighted_nll'] for p in points],
                    marker='o', markersize=3, color=colors[family], alpha=.45, label='Hindsight choices' if r == selected[0] else None)
        ax.axhline(np.mean([r['ordinary_1B_nll'] for r in selected]), color='#243b53', label='Ordinary1B, seed mean')
        ax.axhline(np.mean([r['ordinary_matched_nll'] for r in selected]), color='#243b53', linestyle='--',
                   label='Ordinary1.7B, matched training work')
        ax.set(xlabel='Estimated inference matmuls / ordinary', ylabel='Existing DEV weighted NLL',
               title=family+' — three fitted seeds')
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle('Whole-document oracle: knows targets, pays no selection cost', fontsize=12)
    fig.text(.5, .025, 'Recorded1B models, depths1/2/3 only. Fixed1/3 are untrained exits. Optimistic bound, not a deployed policy.\n'
             'Does not bound token routing, changed hidden trajectories, new training or other architectures.', ha='center', fontsize=8)
    fig.subplots_adjust(bottom=.23, wspace=.25, top=.82)
    for suffix in ['png', 'svg', 'pdf']: fig.savefig(a.output/('oracle-frontier.'+suffix))
    plt.close(fig)
    print(json.dumps([{'family': r['family'], 'seed': r['seed'],
        'oracle_gain_over_compute_matched_ordinary': r['unconstrained_oracle_gain_over_matched_ordinary']}
        for r in records], indent=2))


if __name__ == '__main__':
    main()
