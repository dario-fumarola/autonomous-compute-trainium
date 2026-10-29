import copy
import pytest
from scripts import analyze_joint_exit_probability as a


def fixture():
    # Software-only scalar scores;never training or evaluation source examples.
    losses={'joint-k0':3.2,'joint-k1':3.1,'untied-one-k1':3.15,
      'untied-ordinary-data-k0':3.0,'untied-ordinary-work-k0':2.8,'joint-permuted-k1':4.}
    rows=[]
    for label,loss in losses.items():
        for panel,kind,domain in [('broad','broad','web'),('broad','broad','math'),('broad','broad','code'),('dependency','code_binding','code'),('dependency','math_conclusion','math'),('dependency','math_intermediate_exploratory','math')]:
            for i in range(2):
                rows.append(dict(label=label,panel=panel,kind=kind,domain=domain,id=f'{kind}-{domain}-{i}',group=f'{kind}-{domain}-{i}',targets=2,loss_sum=2*loss,first_loss=loss))
    return rows,dict(bootstrap_seed=7,bootstrap_replicates=32,practical_gain_nats=.01)


def test_supported_improvement_is_not_ordinary_efficiency():
    rows,p=fixture();r=a.analyze(rows,p)
    assert r['screening_gates']==dict(supported_refinement=True,ordinary_equal_data=False,ordinary_equal_work=False)
    assert r['comparisons']['supported_refinement']['broad']['gain']==pytest.approx(.1)
    assert r['comparisons']['ordinary_equal_data']['broad']['gain']==pytest.approx(-.1)
    assert r==a.analyze(rows,p)


@pytest.mark.parametrize('kind',['duplicate','missing','nan','negative_targets'])
def test_invalid_inputs_cannot_make_a_screen_win(kind):
    rows,p=fixture();rows=copy.deepcopy(rows)
    if kind=='duplicate':rows.append(rows[0])
    if kind=='missing':rows.pop(0)
    if kind=='nan':rows[0]['loss_sum']=float('nan')
    if kind=='negative_targets':rows[0]['targets']=-1
    with pytest.raises(AssertionError):a.analyze(rows,p)
