from scripts.run_lean_baseline_tune import project_seconds


def test_complete_tuning_projection_includes_all_rates_and_terminal_audits():
    qualified={'records':{'mixing':{'warm':{'warm_seconds':116.16462083300576}},
                          'ordinary8':{'warm':{'warm_seconds':94.0157756889239}}}}
    result=project_seconds(qualified)
    assert result['parts']['training']==3*576*1.1616462083300576*1.5
    assert result['parts']['cold_starts']==1200
    assert result['parts']['winner_full_gradient_audits']==900
    assert 6000<result['complete_seconds']<7200
