"""Export observed low-depth scores without mixing stable and cooled endpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator


ARMS = {
    'ordinary': ('Ordinary', '#243b53'),
    'fixed1': ('True one-pass', '#007f86'),
    'poisson2': ('Sampled recurrence, exit 2', '#bc4b51'),
}


def load_scores(path):
    raw = path.read_bytes()
    rows = json.loads(raw)['records']
    seen = set()
    for row in rows:
        if row['status'] != 'COMPLETE_INDEPENDENT_LOW_DEPTH_DEVELOPMENT_SCORE':
            raise ValueError('Only completed independent scores can enter figures')
        key = (row['arm'], row.get('readout', row.get('peak_lr')), row['depth'])
        if key in seen:
            raise ValueError('Duplicate plotted score')
        seen.add(key)
        metrics = row['metrics']
        if metrics['weights'] != {'web': .6, 'math': .2, 'code': .2}:
            raise ValueError('Frozen domain weights changed')
        recomputed = sum(metrics['weights'][domain] * values['nll_sum'] / values['real_targets']
                         for domain, values in metrics['domains'].items())
        if (not math.isfinite(recomputed)
                or abs(recomputed - metrics['weighted_nll']) > 1e-10
                or not row['full_state_readback_exact'] or not row['model_unchanged']
                or row['test_accessed']):
            raise ValueError('Score aggregate or independent evaluation scope differs')
    return rows, hashlib.sha256(raw).hexdigest()


def save(fig, output, name):
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'{name}.{suffix}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path,
                        default=Path('results/parcae_low_depth_pilot_20261024'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.results / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 145})
    inputs, table = {}, []
    tune_file = args.results / 'tuning-v2/scores-progress.json'
    if tune_file.exists():
        rows, inputs[str(tune_file)] = load_scores(tune_file)
        if any(r['training_update'] != 576 or r['training_real_targets'] != 301465908
               or r['seed'] != 11 for r in rows):
            raise ValueError('Tuning endpoints must have the same completed schedule/exposure')
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for arm, (label, color) in ARMS.items():
            points = sorted([r for r in rows if r['arm'] == arm], key=lambda r: r['peak_lr'])
            ax.plot([r['peak_lr'] for r in points],
                    [r['metrics']['weighted_nll'] for r in points],
                    marker='o', color=color, label=label)
        ax.set(xscale='log', xlabel='Peak learning rate (AdamW and Muon, 1:1)',
               ylabel='Weighted TUNE NLL (nats per real token)',
               title=f'Small-core rate screen: {len(rows)} of 12 completed scores')
        ax.set_xticks([.002, .004, .008, .016], ['0.002', '0.004', '0.008', '0.016'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(frameon=False)
        fig.text(.5, .035, 'Seed 11; 301.466M natural targets; all points completed decay.\n'
                 'Consumed TUNE selects recipes. Different arms use different work; no seed intervals or efficiency claim.',
                 ha='center', fontsize=8)
        fig.subplots_adjust(left=.12, right=.97, bottom=.21, top=.9)
        save(fig, output, 'tuning')
        table.extend({'phase': 'tuning', 'arm': r['arm'], 'peak_lr': r['peak_lr'],
                      'nll': r['metrics']['weighted_nll'], 'checkpoint_sha256': r['checkpoint_sha256']}
                     for r in rows)

    main_file = args.results / 'main/scores-progress.json'
    if main_file.exists():
        rows, inputs[str(main_file)] = load_scores(main_file)
        points = [r for r in rows if r['depth'] == (2 if r['arm'] == 'poisson2' else 1)]
        if any(r['seed'] != 1231 for r in points):
            raise ValueError('Main pilot has one declared paired seed')
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1))
        for arm, (label, color) in ARMS.items():
            arm_rows = [r for r in points if r['arm'] == arm]
            for axis, key, scale in [(axes[0], 'training_real_targets', 1e9),
                                     (axes[1], 'training_executed_matmuls', 1e18)]:
                stable = sorted([r for r in arm_rows if r['readout'].startswith('stable')],
                                key=lambda r: r[key])
                axis.plot([r[key]/scale for r in stable],
                          [r['metrics']['weighted_nll'] for r in stable],
                          linestyle=':', marker='.', color=color)
                for readout, marker in [('equal', 'o'), ('matched', '*')]:
                    for row in [r for r in arm_rows if r['readout'] == readout]:
                        axis.scatter(row[key]/scale, row['metrics']['weighted_nll'],
                                     marker=marker, s=65 if marker == 'o' else 120,
                                     facecolors='none' if marker == 'o' else color,
                                     edgecolors=color, zorder=4)
        for axis in axes:
            axis.set_ylabel('Weighted DEV NLL (nats per real token)')
        axes[0].set(xlabel='Real source targets (billions)', title='Exposure view')
        axes[1].set(xlabel='Estimated training matrix work (10¹⁸ FLOPs)',
                    title='Complete model paths, including replicated Muon work')
        legend = [Line2D([0], [0], color=color, label=label) for label, color in ARMS.values()]
        legend += [Line2D([0], [0], linestyle=':', marker='.', color='#666666', label='Stable LR'),
                   Line2D([0], [0], linestyle='', marker='o', markerfacecolor='none',
                          color='#666666', label='Cooled equal-data endpoint'),
                   Line2D([0], [0], linestyle='', marker='*', color='#666666',
                          markersize=10, label='Cooled matched-work endpoint')]
        fig.legend(handles=legend, loc='lower center', ncol=3, frameon=False, fontsize=8)
        fig.suptitle('Small-core natural-text pretraining: observed independent FP32 learning curves')
        fig.text(.5, .17, 'One paired seed 1231; consumed DEV; partial scores remain visible. '
                 'Cooled endpoints are not connected to stable curves.\n'
                 'Each endpoint has its own decay. Model-path work excludes separate branches, '
                 'tuning and diagnostics, charged in the campaign ledger.', ha='center', fontsize=8)
        fig.subplots_adjust(left=.07, right=.98, bottom=.30, top=.85, wspace=.27)
        save(fig, output, 'learning-curves')
        table.extend({'phase': 'main', 'arm': r['arm'], 'readout': r['readout'],
                      'depth': r['depth'], 'targets': r['training_real_targets'],
                      'training_matmuls': r['training_executed_matmuls'],
                      'nll': r['metrics']['weighted_nll'], 'checkpoint_sha256': r['checkpoint_sha256']}
                     for r in points)
    if not table:
        raise ValueError('No completed independent score files are available')
    (output / 'plotted-data.json').write_text(json.dumps({
        'input_sha256': inputs, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rows': table, 'scope': 'Completed independent consumed-TUNE/DEV readouts only. '
        'No interpolation across cooldowns, unobserved results, training-seed significance or fresh confirmation.'},
        indent=2) + '\n')
    print(json.dumps({'plotted_points': len(table), 'inputs': inputs, 'assigned_chips': 0}))


if __name__ == '__main__':
    main()
