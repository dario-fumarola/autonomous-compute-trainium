import json
import pytest

from scripts.analyze_chunked_completion import select_available
from scripts.complete_chunked_controls import nonfinite_failure


def test_incomplete_grid_never_brackets_even_when_middle_lr_wins():
    result=select_available([dict(lr=.0003,broad_nll=4),dict(lr=.001,broad_nll=3)],[.0003,.001,.003])
    assert result['lr']==.001 and not result['bracketed'] and not result['complete_numerical_grid']


def test_complete_grid_and_ties():
    rows=[dict(lr=.0003,broad_nll=4),dict(lr=.001,broad_nll=3),dict(lr=.003,broad_nll=5)]
    assert select_available(rows,[.0003,.001,.003])['bracketed']
    rows[0]['broad_nll']=3
    assert not select_available(rows,[.0003,.001,.003])['bracketed']


@pytest.mark.parametrize('rows',[[],[dict(lr=.001,broad_nll=float('inf'))],
    [dict(lr=.001,broad_nll=3),dict(lr=.001,broad_nll=4)],[dict(lr=.002,broad_nll=3)]])
def test_rejects_fabricated_or_unregistered_candidates(rows):
    with pytest.raises(AssertionError):select_available(rows,[.0003,.001,.003])


def test_only_explicit_gradient_nonfinite_is_a_completed_failed_opportunity(tmp_path):
    p=tmp_path/'snapshots'/'phase';p.mkdir(parents=True);logs=tmp_path/'logs';logs.mkdir()
    (p/'closure.json').write_text(json.dumps(dict(exit_code=1)))
    supervisor=dict(worker_exit_code=1,supervisor_failure=None,container_oom_kill_delta=0)
    (logs/'phase.supervisor.json').write_text(json.dumps(supervisor))
    (logs/'phase.log').write_text('The total norm of order 2.0 for gradients is non-finite, so it cannot be clipped')
    assert nonfinite_failure(tmp_path,'phase')
    supervisor['container_oom_kill_delta']=1
    (logs/'phase.supervisor.json').write_text(json.dumps(supervisor))
    assert not nonfinite_failure(tmp_path,'phase')
