"""Learning curves from completed independent CPU DEV readouts only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=Path('results/mature_fp32_20261024'))
    p.add_argument('--parents', type=Path,
                   default=Path('results/mature_fp32_parent_audit_20261024/common-fp32-parents.json'))
    a = p.parse_args()
    parent = json.loads(a.parents.read_text())['rows']
    audit = a.results/'independent-cpu-verification.json'
    if not audit.exists():
        audit = a.results/'live-independent-cpu.json'
    current = json.loads(audit.read_text())['records']
    plan = json.loads((a.results/'plan.json').read_text())
    rows = []
    for raw in parent+current:
        update = raw.get('update')
        if update is None:
            update = 7680 if raw in parent else int(raw['identity'].rsplit('-u', 1)[1])
        rows.append({'arm': raw['arm'], 'update': update,
                     'real_targets': raw['training_real_targets'],
                     'phase': 'completed decay' if update == 11520 else 'stable',
                     'cpu_fp32_nll': raw['metrics']['weighted_nll'],
                     'domains': {k: v['nll'] for k, v in raw['metrics']['domains'].items()},
                     'training_matmul_flops': update*plan['step_flops'][raw['arm']],
                     'checkpoint_sha256': raw['checkpoint_sha256']})
    updates = sorted({r['update'] for r in rows})
    pairs = []
    for update in updates:
        pair = {r['arm']: r for r in rows if r['update'] == update}
        if set(pair) != {'ordinary', 'fixed'}:
            continue
        assert pair['ordinary']['real_targets'] == pair['fixed']['real_targets']
        pairs.append({'update': update, 'real_targets': pair['ordinary']['real_targets'],
                      'phase': pair['ordinary']['phase'],
                      'equal_data_gain': pair['ordinary']['cpu_fp32_nll']-pair['fixed']['cpu_fp32_nll']})
    out = a.results/'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 145})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = {'ordinary': '#243b53', 'fixed': '#bc4b51'}
    for arm in colors:
        points = sorted([r for r in rows if r['arm'] == arm], key=lambda r: r['update'])
        stable = [r for r in points if r['phase'] == 'stable']
        axes[0].plot([r['real_targets']/1e9 for r in stable], [r['cpu_fp32_nll'] for r in stable],
                     marker='o', color=colors[arm], label='Ordinary' if arm == 'ordinary' else 'Fixed two passes')
        for end in [r for r in points if r['phase'] == 'completed decay']:
            before = stable[-1]
            axes[0].plot([before['real_targets']/1e9, end['real_targets']/1e9],
                         [before['cpu_fp32_nll'], end['cpu_fp32_nll']], ':', color=colors[arm])
            axes[0].scatter(end['real_targets']/1e9, end['cpu_fp32_nll'], marker='D', color=colors[arm])
    axes[0].set(xlabel='Real source targets (billions)', ylabel='Weighted natural-text DEV NLL',
                title='Independent FP32 scoring, one paired seed')
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].axhline(0, color='#777777', linewidth=.8)
    stable = [r for r in pairs if r['phase'] == 'stable']
    axes[1].plot([r['real_targets']/1e9 for r in stable], [r['equal_data_gain'] for r in stable],
                 marker='o', color=colors['fixed'])
    for end in [r for r in pairs if r['phase'] == 'completed decay']:
        before = stable[-1]
        axes[1].plot([before['real_targets']/1e9, end['real_targets']/1e9],
                     [before['equal_data_gain'], end['equal_data_gain']], ':', color=colors['fixed'])
        axes[1].scatter(end['real_targets']/1e9, end['equal_data_gain'], marker='D', color=colors['fixed'])
    axes[1].set(xlabel='Real source targets (billions)', ylabel='Ordinary NLL minus fixed NLL',
                title='Positive: fixed improves at equal data')
    fig.suptitle('150M natural-text pretraining: paired precision continuation', fontsize=13)
    fig.text(.5, .045,
             'FP32 training starts at2.010B after BF16 history. Circles: stable LR; diamonds: completed decay.\n'
             'One reused seed, consumed DEV; no seed intervals. Fixed uses1.692× estimated matmuls. No efficiency claim.',
             ha='center', fontsize=8)
    fig.subplots_adjust(left=.08, right=.98, bottom=.22, top=.83, wspace=.34)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(out/('paired-learning-curve.'+suffix))
    plt.close(fig)
    (out/'learning-curve-data.json').write_text(json.dumps({'rows': rows, 'paired_points': pairs,
        'scope': 'Only completed independent CPU FP32 existing-DEV scores; unpaired partial points excluded from gain curve.'}, indent=2)+'\n')
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
    main()
