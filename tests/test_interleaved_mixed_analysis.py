import argparse
import json
from pathlib import Path
from scripts.analyze_interleaved_mixed_tuning import main


def test_mixed_analysis_selects_only_registered_exit_and_retains_other_exit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    identity='fixture'; folder=tmp_path/'runs'/(identity+'-evaluation'); folder.mkdir(parents=True)
    (folder/'report.json').write_text(json.dumps({'status':'COMPLETE_INTERLEAVED_SCREEN_EVALUATION'}))
    grid=[.0003,.001,.003]
    plan={'identity':identity,'learning_rate_grid':grid,'arms':{f'mixed-lr{i}':{'lr':lr} for i,lr in enumerate(grid)}}
    planpath=tmp_path/'plan.json'; planpath.write_text(json.dumps(plan))
    rows=[]
    for i,lr in enumerate(grid):
        for suffix,value in [('main',[.9,.7,.8][i]),('supported3',[.6,.5,.4][i])]:
            for j in range(384):
                rows.append(dict(label=f'mixed-lr{i}-{suffix}',panel='broad',id=str(j),
                    domain=('web','math','code')[j//128],loss_sum=value*512,targets=512))
    for rank in range(64):
        (folder/f'PRIVATE-rank{rank}-scores.json').write_text(json.dumps(rows[rank::64]))
    old=Path('results/interleaved_workspace_20261029/tuning-v1'); old.mkdir(parents=True)
    (old/'analysis.json').write_text(json.dumps({'selection':{
        'ordinary':{'broad_nll':.73},'pause':{'broad_nll':.72},'latent':{'broad_nll':.71}}}))
    output=tmp_path/'analysis'
    main(argparse.Namespace(root=tmp_path,plan=planpath,output=output))
    result=json.loads((output/'report.json').read_text())
    assert result['selection']['arm']=='mixed-lr1'
    assert result['selection']['bracketed']
    assert result['scores']['mixed-lr2']['exits']['supported3']['broad_nll'] < .41
    assert result['prediction_screen_pass']
    assert result['protected_groups_scored']==0
