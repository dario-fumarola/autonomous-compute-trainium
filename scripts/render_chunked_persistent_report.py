"""Render completed frozen results; no fitting, selection or new evaluation."""
import argparse
import json
import math
from pathlib import Path

from compute_trainium.training import file_sha256


IDENTITY = 'chunked-persistent-screen-v1'
PUBLIC = Path('results/chunked_persistent_workspace_20261029/learning-v1')
FAMILIES = ('ordinary', 'generated', 'constant')
LABELS = dict(ordinary='Ordinary', generated='Generated workspace', constant='Constant workspace')


def read(path):
    return json.loads(path.read_text())


def render(root, *, completion=False):
    identity = 'chunked-control-completion-v1' if completion else IDENTITY
    public = PUBLIC.parent / 'control-completion-v1' if completion else PUBLIC
    folder = root / identity
    result = read(folder / 'report.json')
    expected = 'COMPLETE_CHUNKED_CONTROL_COMPLETION' if completion else 'COMPLETE_CHUNKED_PERSISTENT_SCREEN'
    if result['status'] != expected:
        raise RuntimeError('No report rendering before the frozen screen completes')
    plan = read(folder / ('readout-plan.json' if completion else 'plan.json'))
    analysis = read(folder / 'compare/report.json')
    selection = read(folder / 'select/report.json')
    exact = read(folder / 'readout/report.json')
    cost_files = ([PUBLIC / 'execution-costs.json', public / 'fit-execution-costs.json',
                   public / 'readout-execution-costs.json'] if completion else [PUBLIC / 'execution-costs.json'])
    costs = dict(native_attempts=[r for f in cost_files for r in read(f)['native_attempts']])
    if any(file_sha256(path) != sha for path, sha in plan['source_sha256'].items()):
        raise AssertionError('Original frozen scientific sources changed')
    for arm in plan['arms']:
        if result['records']['audit-' + arm]['status'] != 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':
            raise AssertionError('Every completed endpoint must pass its audit')
    receipts = [read(PUBLIC.parent / name / 'execution-costs.json') for name in
                ('numerical-v1', 'numerical-recovery-v1', 'completion-v1')]
    attempts = [row for receipt in [*receipts, costs] for row in receipt['native_attempts']]
    if len({row['identity'] for row in attempts}) != len(attempts):
        raise AssertionError('Cost histories must not be double-counted')
    qualification_hours = sum(row['assigned_chip_hours'] for receipt in receipts
                              for row in receipt['native_attempts'])
    screen_hours = sum(row['assigned_chip_hours'] for row in costs['native_attempts'])
    if not all(math.isfinite(x) and x >= 0 for x in (qualification_hours, screen_hours)):
        raise AssertionError('Finite closed cost receipts required')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    palette = dict(ordinary='#505b68', generated='#007f78', constant='#b96a16')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, layout='constrained')
    for i, lr in enumerate(plan['learning_rate_grid']):
        ax = axes[i]
        missing = []
        for family in FAMILIES:
            if f'{family}-lr{i}' not in plan['arms']:
                missing.append(LABELS[family])
                continue
            arm = plan['arms'][f'{family}-lr{i}']
            updates = read(root / 'runs' / arm['phase'] / 'updates.json')
            if len(updates) != len(arm['depth_schedule']) or updates[-1]['update'] != len(updates):
                raise AssertionError('All original training updates required')
            weight = np.array([row['real_targets'] for row in updates], dtype=float)
            loss = np.array([row['loss'] for row in updates], dtype=float)
            smooth = np.convolve(loss * weight, np.ones(64), 'valid') / np.convolve(weight, np.ones(64), 'valid')
            work = np.array([row['cumulative_matrix_flops'] for row in updates[63:]], dtype=float) / 1e16
            ax.plot(work, smooth, label=LABELS[family], color=palette[family], linewidth=1.3)
        ax.set_title(f'Peak LR {lr:g}')
        if missing:
            ax.text(.98,.96,'Unavailable endpoint:\n'+'\n'.join(missing),transform=ax.transAxes,
                    ha='right',va='top',fontsize=8,color='#854040')
        ax.set_xlabel('Nominal training matrix FLOPs / 10¹⁶')
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Training NLL, 64-update target-weighted mean')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('All complete fits — training trajectories, not held-out scores')
    fig.savefig(public / 'training-curves.png', dpi=180)
    fig.savefig(public / 'training-curves.svg')
    plt.close(fig)

    opening = (f"{len(plan['arms'])} of nine original learning-rate opportunities completed with mature audits. "
               'The original screen remains STOPPED after a non-finite-gradient attempt. '
               'This separate completion trains only the three previously unstarted controls and reuses the five audited histories. '
               'No failed fit is retried, and no LR, endpoint or original deadline is extended. '
               'The original all-nine-complete gate stays UNMET; unavailable fits have no invented NLL. '
               if completion else 'All nine registered fits and mature numerical audits completed. ')
    lines = ['# Sparse persistent workspace: ' + ('descriptive control completion' if completion else 'complete matched-work screen'), '',
             opening + 'Each family received three independent learning-rate opportunities. '
             'The selected checkpoints were rescored in FP64 on all 384 reused DEV sources. '
             'This is one paired random-initialization seed, with 60/20/20 web/math/code weighting. '
             'Both fresh evaluation reserves remain unscored; terminal alpha spent is zero.', '',
             '| Selected family | Peak LR | Exact broad NLL ↓ | Web | Math | Code |',
             '|---|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        s = analysis['scores'][family]
        d = s['domains']
        lines.append(f"| {LABELS[family]} | {analysis['selection'][family]['lr']:g} | {s['broad_nll']:.9f} | {d['web']:.9f} | {d['math']:.9f} | {d['code']:.9f} |")
    lines += ['', 'Positive gain favors the generated workspace.', '',
              '| Comparator | Generated gain, nats | Conditional 95% interval | Registered development gate |',
              '|---|---:|---|---|']
    for family in ('ordinary', 'constant'):
        key = 'generated_over_' + family
        c = analysis['comparisons'][key]['comparison']
        lo, hi = c['conditional_95_interval']
        lines.append(f"| {LABELS[family]} | {c['gain']:+.9f} | [{lo:+.9f}, {hi:+.9f}] | {'PASS' if analysis['decisions'][key] else 'FAIL'} |")
    lines += ['', 'Each gate requires a gain of at least 0.01 nats, a positive conditional lower bound, '
              'and interior learning-rate optima for all three families. These intervals condition on '
              'selected models and reused development sources. They do not measure training-seed uncertainty '
              'or constitute confirmation.', '',
              '**Both registered gates: ' + ('PASS' if analysis['both_development_gates_pass'] else 'FAIL') + '.** '
              'No automatic scaling or confirmation follows. Neither outcome establishes learned allocation, '
              'transferable reasoning or useful additional inference steps.', '',
              '## All learning-rate opportunities', '',
              'These are the common accelerator selection readouts; the selected-model table above uses the '
              'independent FP64 readout. No endpoint is omitted.', '',
              '| Family | Peak LR | Selection NLL ↓ | Selected |', '|---|---:|---:|---|']
    for family in FAMILIES:
        for row in selection['all_scores'][family]:
            chosen = row['arm'] == analysis['selection'][family]['arm']
            lines.append(f"| {LABELS[family]} | {row['lr']:g} | {row['broad_nll']:.9f} | {'yes' if chosen else ''} |")
    if completion:
        for arm, failure in analysis['unavailable'].items():
            family=arm.split('-')[0]
            lines.append(f"| {LABELS[family]} | {failure['lr']:g} | unavailable: numerical failure | no |")
    lines += ['', '## Resources', '',
              '| Selected family | Parameters | Natural targets | Nominal training matrix FLOPs | Warm update, s | Worker fit, s |',
              '|---|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        r = analysis['resources'][family]
        lines.append(f"| {LABELS[family]} | {r['parameters']:,} | {r['real_targets']:,} | {r['nominal_training_matrix_flops']:,} | {r['mean_warm_update_seconds']:.6f} | {r['wall_seconds']:.3f} |")
    lines += ['', f"Ordinary receives {analysis['ordinary_work_ratio']:.9f} times the generated arm's nominal training matrix work. "
              'This ledger counts executed matrix rectangles, including the vocabulary projection, '
              'with FMA=2 and backward=2× forward. Elementwise work, communication and launch latency '
              'are outside it. Worker fit durations include startup, compilation and checkpoint work; '
              'warm-update durations exclude startup. These are descriptive cross-run timings, not a hardware-time-matched trial.', '',
              f'All original and completion attempts plus accelerator readout used **{screen_hours:.6f} assigned chip-hours**. '
              f'Its preceding qualifications used **{qualification_hours:.6f} assigned chip-hours**, including both timed-out attempts. '
              f'Total assigned accelerator cost for these histories is **{screen_hours + qualification_hours:.6f} chip-hours**. '
              'Host-only audits and readouts are recorded separately. Reserved-instance opportunity cost is not added again to assigned chip-time.', '',
              '## Numerical checks and scope', '',
              'Each scored endpoint passed the unchanged independent full-global TRAIN-gradient audit, full model/optimizer/RNG '
              'restoration and replica checks. The selected exact readout contains '
              f"{sum(len(w['checks']) for w in exact['workers'])} cached/full-prefix checks. "
              'All genuine targets are scored once with uniform next-token weighting. Generated and constant slots '
              'have identical parameters and matrix work; the constant-input body can still read its causal prefix.', '',
              ('The original position diagnostic is not admitted because the original completion prerequisite failed. '
               if completion else 'The separately registered position diagnostic decomposes this same broad result. Its bins cannot repair a failed broad gate. ')
              + 'Unsupported slot budgets, fresh confirmation and serving performance '
              'have not been evaluated.', '',
              '![All training trajectories](training-curves.png)', '',
              'Machine-readable evidence: [analysis](compare.json), [all selections](select.json), '
              '[exact readout](readout.json), '+
              ('[original costs](../learning-v1/execution-costs.json), [remaining-fit costs](fit-execution-costs.json), [readout costs](readout-execution-costs.json), '
               if completion else '[execution costs](execution-costs.json), ')+ '[full report](report.json).', '']
    (public / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--completion', action='store_true')
    args=parser.parse_args()
    render(args.root,completion=args.completion)
