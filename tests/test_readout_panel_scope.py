import json
import pytest
from compute_trainium.readout_panel_scope import verify_train_panel,mark_train_report


def test_train_reuse_requires_explicit_role_and_cannot_be_marked_heldout(tmp_path):
    plan=tmp_path/'plan.json';output=tmp_path/'output';output.mkdir();report=output/'report.json'
    plan.write_text(json.dumps(dict(panel_role='dev')));report.write_text(json.dumps(dict(status='PASS_INDEPENDENT_ORIGINAL_CHAIN_READOUT',scope='old scope')))
    args=['--plan',str(plan),'--output',str(output)]
    with pytest.raises(ValueError):verify_train_panel(args)
    plan.write_text(json.dumps(dict(panel_role='train_only_post_pilot_diagnostic')));verify_train_panel(args);mark_train_report(args)
    saved=json.loads(report.read_text());assert saved['panel_role']=='train_only_post_pilot_diagnostic' and saved['heldout_result'] is False and 'TRAIN-only' in saved['scope']
