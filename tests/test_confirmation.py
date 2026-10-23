import pytest
from compute_trainium.confirmation import matched_endpoint, paired_seed_interval, endpoint_inventory


def test_discrete_compute_control_never_overspends_and_is_maximal():
    plan = matched_endpoint(3840, 483956914913280, 286044821913600)
    assert plan['updates'] == 6496
    assert plan['stable_update'] == 5846
    assert plan['candidate_flops'] <= plan['reference_flops']
    assert plan['candidate_flops'] + 286044821913600 > plan['reference_flops']


def test_fresh_pair_uses_df_one_not_document_count():
    result = paired_seed_interval({'47': .02, '71': .04})
    assert result['interval_95'] == pytest.approx([.03-.12706204736432095, .03+.12706204736432095])
    assert result['decision'] == 'INCONCLUSIVE'
    assert paired_seed_interval({'23': -.03, '47': -.03, '71': -.03})['decision'] == 'USEFUL_GAIN_RULED_OUT_AT_THIS_EXPOSURE'
    with pytest.raises(ValueError):
        paired_seed_interval({'23': .02})


def test_confirmation_inventory_contains_only_twelve_declared_trained_paths():
    models=endpoint_inventory('/work/instance-nvme')
    assert len(models)==len({m['identity'] for m in models})==12
    assert {m['seed'] for m in models if m['arm']=='untied'}=={23,47,71}
    assert all(m['passes']==(1 if m['arm']=='ordinary' else 2) for m in models)
    assert next(m for m in models if m['identity']=='ordinary-flop-s23')['checkpoint'].endswith('u6496/checkpoint.pt')


def test_campaign_fails_closed_before_any_device_or_test_access(tmp_path,monkeypatch):
    import json
    import sys
    from scripts.run_confirmation_campaign import main
    campaign=tmp_path/'old';(campaign/'queue').mkdir(parents=True)
    (campaign/'queue/report.json').write_text(json.dumps({'status':'FAILED_STAGE'}))
    output=tmp_path/'confirmation'
    monkeypatch.setenv('RESEARCH_ROOT',str(tmp_path));monkeypatch.setenv('RESEARCH_CONTAINER_ROOT','/work')
    monkeypatch.setattr(sys,'argv',['run_confirmation_campaign','--campaign',str(campaign),
        '--output',str(output),'--absolute-deadline-unix','9999999999'])
    main()
    assert json.loads((output/'queue/report.json').read_text())['status']=='CONFIRMATION_BLOCKED_ORIGINAL_CAMPAIGN'
    assert not (output/'test-release.json').exists()
    assert not (output/'followups').exists()


def test_full_confirmation_analysis_keeps_seed_and_source_uncertainty_separate(tmp_path,monkeypatch):
    """Numerical score fixtures only; never reads a research corpus or checkpoint."""
    import json
    import sys
    from dataclasses import asdict
    from compute_trainium.metrics import DocumentScore,summarize
    from scripts.analyze_frozen_test import main
    root=tmp_path/'private';campaign=tmp_path/'campaign';campaign.mkdir()
    models=endpoint_inventory('/work/new')
    for m in models:
        m['run']='fixture-'+m['identity'];out=root/'runs'/m['run'];out.mkdir(parents=True)
        nll=2+m['seed']/10000+{'ordinary':0,'shared':.03,'untied':-.02}[m['arm']]
        if m['identity']=='ordinary-flop-s23':nll-=.04
        scores=[DocumentScore(domain,f'{domain}-{g}',f'group{g}',100*(nll+i*.1),100)
            for g in range(3) for i,domain in enumerate(['web','math','code'])]
        (out/'document-scores.json').write_text(json.dumps([asdict(s) for s in scores]))
        (out/'report.json').write_text(json.dumps({'status':'PASS_SHARDED_TRAINED_PATH_READOUT',
            'split':'test','metrics':summarize(scores),'training_real_tokens':1234}))
    (campaign/'test-release.json').write_text(json.dumps({'models':models}))
    output=tmp_path/'report'
    monkeypatch.setattr(sys,'argv',['analyze_frozen_test','--root',str(root),
        '--campaign',str(campaign),'--output',str(output)])
    main();report=json.loads((output/'summary.json').read_text())
    assert report['three_seed_1b_confirmation']['mean_gain_nats']==pytest.approx(-.03)
    assert report['three_seed_1b_confirmation']['decision']=='USEFUL_GAIN_RULED_OUT_AT_THIS_EXPOSURE'
    fresh=report['paired_seed_comparisons']['ordinary_to_untied_fresh_seeds']
    assert fresh['degrees_of_freedom']==1 and set(fresh['paired_seed_gains'])=={'47','71'}
    assert fresh['mean_gain_nats']==pytest.approx(.02)
    resource=report['single_pair_diagnostics']['ordinary_matched_flops_to_untied_1b']
    assert resource['gain_nats']==pytest.approx(-.02)
    assert 'Within-seed' in resource['warning']
