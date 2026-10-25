import copy

import pytest

from compute_trainium.future_span_context import context_cells


def fixture():
    # Bookkeeping loss fixtures only;never research text or model examples.
    cells = {k: [] for k in ['one-ntp', 'one-future', 'recurrent-ntp', 'recurrent-future']}
    for group in range(8):
        for band in ['129-256', '257-512', '513-900']:
            common = {'id': f'{group}-{band}', 'group': str(group), 'band': band, 'real_targets': 1}
            for cell in ['one-ntp', 'one-future']:
                cells[cell].append({**common, 'nll_sum': {'full': 1., 'short': 2.}})
            cells['recurrent-ntp'].append({**common,
                'nll_sum': {'full1': 1., 'full2': .8, 'short1': 2., 'short2': 1.9}})
            cells['recurrent-future'].append({**common,
                'nll_sum': {'full1': 1., 'full2': .7, 'short1': 2., 'short2': 1.9}})
    return cells


def test_objective_interaction_and_absolute_context_gains_keep_their_signs():
    result = context_cells(fixture(), draws=100)
    assert result['future_objective_interaction_on_full_context']['gain_nats'] == pytest.approx(.1)
    assert all(v['exploratory_joint_signs_only'] for v in result['by_objective'].values())
    assert not result['terminal_success'] and not result['overrides_primary_nll_decision']
    bad = fixture()
    for r in bad['recurrent-future']:
        # Positive interaction achieved entirely by harming the short view.
        r['nll_sum'].update(full2=1.1, short2=3.)
    result = context_cells(bad, draws=100)
    assert result['by_objective']['future']['within_model']['interaction'] > 0
    assert not result['by_objective']['future']['exploratory_joint_signs_only']


def test_all_cells_and_original_source_target_pairing_are_required():
    cells = fixture()
    del cells['one-future']
    with pytest.raises(ValueError, match='four'):
        context_cells(cells, draws=100)
    cells = fixture()
    cells['recurrent-future'][0]['group'] = 'different-source'
    with pytest.raises(ValueError, match='pair'):
        context_cells(cells, draws=100)
    cells = fixture()
    cells['one-ntp'][0]['nll_sum']['full2'] = .1
    with pytest.raises(ValueError, match='supported'):
        context_cells(cells, draws=100)
