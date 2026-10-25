"""Four-cell exploratory contrasts on the unchanged natural identifier panel."""
import math

from .code_context_control import BAND_WEIGHTS, compare_trained_control
from .code_context_views import summarize, group_intervals
from .metrics import DocumentScore, paired_group_interval


def context_cells(cells, *, draws=5000, seed=1381):
    expected = {'one-ntp', 'one-future', 'recurrent-ntp', 'recurrent-future'}
    if set(cells) != expected:
        raise ValueError('All four objective/architecture cells are required')
    paired = {}
    for cell, rows in cells.items():
        names = {'full', 'short'} if cell.startswith('one') else {'full1', 'full2', 'short1', 'short2'}
        targets = {}
        for row in rows:
            if (row['id'] in targets or row['real_targets'] < 1
                    or set(row['nll_sum']) != names
                    or any(not math.isfinite(v) for v in row['nll_sum'].values())):
                raise ValueError('Unique finite supported-path identifier scores required')
            targets[row['id']] = (row['group'], row['band'], row['real_targets'])
        paired[cell] = targets
    if any(paired[cell] != paired['one-ntp'] for cell in expected):
        raise ValueError('Original targets, repository groups, bands and token counts must pair')
    by_objective = {}
    for objective in ['ntp', 'future']:
        control = cells['one-' + objective]
        recurrent = cells['recurrent-' + objective]
        within = group_intervals(recurrent, draws=draws, seed=seed)
        comparison = compare_trained_control(control, recurrent, draws=draws, seed=seed)
        comparison['scope'] = ('Same-objective trained one-pass control in this four-cell exploratory readout. '
            'Unchanged consumed targets and repository intervals;not independent training-seed confirmation.')
        by_objective[objective] = {'within_model': summarize(recurrent),
            'conditional_repository_uncertainty': within, 'trained_onepass_comparison': comparison,
            'exploratory_joint_signs_only': (
                within['intervals_95']['full_gain'][0] > 0
                and within['intervals_95']['interaction'][0] > 0
                and comparison['candidate_gain_full']['interval_95'][0] > 0)}
    index = {cell: {r['id']: r for r in rows} for cell, rows in cells.items()}
    left, right = [], []
    for key, (group, band, count) in paired['one-ntp'].items():
        # Difference of recurrence advantages, each original target counted once.
        a = index['one-future'][key]['nll_sum']['full'] + index['recurrent-ntp'][key]['nll_sum']['full2']
        b = index['recurrent-future'][key]['nll_sum']['full2'] + index['one-ntp'][key]['nll_sum']['full']
        left.append(DocumentScore(band, key, group, a, count))
        right.append(DocumentScore(band, key, group, b, count))
    interaction = paired_group_interval(left, right, draws=draws, seed=seed, weights=BAND_WEIGHTS)
    return {'by_objective': by_objective, 'future_objective_interaction_on_full_context': interaction,
        'identifier_targets': len(left), 'real_identifier_tokens': sum(r.tokens for r in left),
        'terminal_success': False, 'overrides_primary_nll_decision': False,
        'scope': 'Exploratory consumed-panel context-use readout. Nominal repository uncertainty conditional on one fitted training seed;not training-seed significance,reasoning,terminal alpha or a replacement for pooled NLL/resource controls.'}
