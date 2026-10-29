"""Numerical arrays and admission fixtures, never research examples."""
import copy
import json

import numpy as np
import pytest

from compute_trainium.chunked_position_analysis import make_rows, summarize
import scripts.diagnose_chunked_completion_positions as d


def fixture():
    meta=[dict(id=str(i),group=str(i//2),domain=('web','math','code')[i//128]) for i in range(384)]
    by={f:make_rows(np.full((384,512),value),meta,list(range(384)))
        for f,value in [('ordinary',3.),('generated',2.9),('constant',2.95)]}
    parts=[{f:rows[k::4] for f,rows in by.items()} for k in range(4)]
    return parts,{f:summarize(rows) for f,rows in by.items()}


def test_complete_pairing_all_three_comparisons_and_reconstruction():
    parts,expected=fixture();r=d.aggregate(parts,expected,replicates=10)
    for name,gain in [('generated_over_ordinary',.1),('generated_over_constant',.05),('constant_over_ordinary',.05)]:
        c=r['comparisons'][name]
        assert c['full']['broad']['gain']==pytest.approx(gain)
        assert sum(x['contribution_to_broad_gain'] for x in c['strata'].values())==pytest.approx(gain)


def test_domain_mismatch_cannot_hide_behind_matching_broad_score():
    parts,expected=fixture();bad=copy.deepcopy(expected)
    bad['ordinary']['domains']['math']+=.01
    with pytest.raises(AssertionError,match='Every domain'): d.aggregate(parts,bad,replicates=10)
    parts[0]['ordinary'].pop()
    with pytest.raises(AssertionError): d.aggregate(parts,expected,replicates=10)


@pytest.mark.parametrize('artifact',['progress.json','PRIVATE-rank0-scores.json'])
def test_any_existing_quality_artifact_blocks_new_registration(tmp_path,artifact):
    source=tmp_path/d.SCREEN;source.mkdir();(source/'plan.json').write_text('{}')
    evaluation=tmp_path/'runs'/(d.SCREEN+'-evaluation');evaluation.mkdir(parents=True)
    (evaluation/artifact).write_text('{}')
    with pytest.raises(AssertionError,match='quality artifacts'):d.prepare(tmp_path)
    assert not (tmp_path/d.ID).exists()
