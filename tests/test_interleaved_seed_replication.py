import copy
import json
import pytest
from scripts.run_interleaved_seed_replication import SEEDS,one_trial,select_recipes


def test_trial_changes_only_seed_identity_not_selected_scientific_recipe():
    parent=dict(seed=9901,model_config={'layers':6,'dim':512},data_report_sha256='data',
        arms={'old':dict(lr=.001,depth_schedule=[2,3,2],real_targets=120,warmup_updates=1,
                         decay_start=2,checkpoint_updates=[3],audit_depths=[2,3],phase='old')})
    original=copy.deepcopy(parent)
    p=one_trial(parent,'latent',SEEDS[0],'new',{'arm':'old'})
    assert parent==original
    assert p['seed']==SEEDS[0] and p['model_config']==parent['model_config']
    for key in ('lr','depth_schedule','real_targets','warmup_updates','decay_start','checkpoint_updates','audit_depths'):
        assert p['arms']['latent'][key]==parent['arms']['old'][key]
    assert p['arms']['latent']['phase']=='new-fit'
    assert p['learning_rate_grid']==[.001] and not p['automatic_promotion']
    with pytest.raises(ValueError):one_trial(parent,'latent',9901,'bad',{'arm':'old'})


def test_control_selection_uses_best_all_registered_shapes_and_fails_closed(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);base=tmp_path/'results/interleaved_workspace_20261029'
    def write(name,r):
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r))
    def selection(arm,score):return dict(arm=arm,lr=.001,broad_nll=score,bracketed=True)
    write('tuning-v1/analysis.json',dict(selection={k:selection(k,4.) for k in ('ordinary','pause')}))
    write('mixed-round-tuning-v1/analysis.json',dict(selection=selection('mixed',3.)))
    for label,score in [('tied',3.1),('frontier-deeper',3.2),('frontier-wider',3.15)]:
        status='COMPLETE_TIED_ORDINARY_PAIRED_COMPARISON' if label=='tied' else 'COMPLETE_ORDINARY_FRONTIER_PAIRED_COMPARISON'
        write(label+'-comparison-v1/report.json',dict(status=status,useful_equal_data_screen=True,selected_control=selection(label,score)))
    recipes,gates=select_recipes(tmp_path)
    assert recipes['ordinary']['tied'] and len(gates)==3
    path=base/'frontier-wider-comparison-v1/report.json';r=json.loads(path.read_text());r['selected_control']['broad_nll']=3.05;path.write_text(json.dumps(r))
    recipes,_=select_recipes(tmp_path)
    assert recipes['ordinary']['recipe']=='interleaved-frontier-wider-tuning-v1' and not recipes['ordinary']['tied']
    r['useful_equal_data_screen']=False;path.write_text(json.dumps(r))
    with pytest.raises(AssertionError):select_recipes(tmp_path)
