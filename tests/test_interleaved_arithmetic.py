import numpy as np
import pytest

from scripts.readout_interleaved_arithmetic import view, analysis, LABELS, KINDS


def test_short_view_preserves_every_original_answer_and_teacher_forcing():
    x = np.arange(80)
    y = x+1
    mask = np.zeros(80,dtype=bool)
    mask[50:54] = True
    full, yf, pf = view(x,y,mask,'full')
    short, ys, ps = view(x,y,mask,'last32')
    np.testing.assert_array_equal(yf,ys)
    np.testing.assert_array_equal(full[19:],short)
    assert pf == [50,51,52,53] and ps == [31,32,33,34]
    np.testing.assert_array_equal(short[np.array(ps[1:])],ys[:-1])
    with pytest.raises(ValueError):
        view(x,y,np.zeros(80,dtype=bool),'full')
    mask[52] = False
    with pytest.raises(ValueError):
        view(x,y,mask,'full')


def test_paired_analysis_preserves_sign_groups_and_all_controls():
    rows = []
    for label in LABELS:
        for condition in ('full','last32'):
            for kind in KINDS:
                for group in range(4):
                    loss = 2 + .1*(label!='latent') + .2*(condition=='last32')
                    rows.append(dict(label=label,condition=condition,kind=kind,id=f'{kind}-{group}',
                        group=str(group),targets=2,loss_sum=2*loss,first_loss=loss,correct_tokens=1,whole_correct=0))
    result = analysis(rows,replicates=100)
    assert all(result['conclusion_gates'].values())
    assert result['comparisons']['math_conclusion']['full_vs_ordinary']['gain'] == pytest.approx(.1)
    assert result['comparisons']['math_conclusion']['context_utility_latent']['gain'] == pytest.approx(.2)
    with pytest.raises(AssertionError):
        analysis(rows[:-1],replicates=100)
