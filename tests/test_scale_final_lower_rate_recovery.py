import copy
import json
import pytest

from compute_trainium.training import file_sha256
from scripts import run_scale_boundary_lower_rates as selection
from scripts import run_scale_final_lower_rate_recovery as recovery


def completed():
    return dict(audit=dict(status='PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT',
        full512=dict(windows=512,gradient=dict(passes_two_percent_gate=True)),replay=dict(passes=True)),
        score=dict(status='COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT'),cleanup_receipt_sha256='fixture')


def stopped(root):
    order=['fixed-two-lower-lr0','ordinary8-lower-lr0','fixed-two-lower-lr1',recovery.CASE]
    return dict(case_order=order),dict(status='STOPPED_BOUNDARY_LOWER_RATES',
        error='Filesystem inode reserve failed: '+str(root),
        records={k:completed() if k!=recovery.CASE else {} for k in order})


def test_recovery_accepts_only_the_last_incomplete_rate(tmp_path):
    parent,value=stopped(tmp_path);recovery.validate_stop(tmp_path,parent,value)
    bad=copy.deepcopy(value);bad['records'][recovery.CASE]={'score':{}}
    with pytest.raises(ValueError):recovery.validate_stop(tmp_path,parent,bad)
    bad=copy.deepcopy(value);bad['error']='Numerical audit failed'
    with pytest.raises(ValueError):recovery.validate_stop(tmp_path,parent,bad)


def test_recovery_never_skips_an_earlier_failed_audit(tmp_path):
    parent,value=stopped(tmp_path)
    value['records']['fixed-two-lower-lr1']['audit']['full512']['gradient']['passes_two_percent_gate']=False
    with pytest.raises(ValueError):recovery.validate_stop(tmp_path,parent,value)


def recovery_files(root):
    original=root/selection.IDENTITY;original.mkdir();plan=original/'plan.json';plan.write_text('{}')
    _,value=stopped(root);value['plan_sha256']=file_sha256(plan)
    report=original/'report.json';report.write_text(json.dumps(value))
    folder=root/selection.FINAL_RECOVERY;folder.mkdir();next_plan=folder/'plan.json'
    next_plan.write_text(json.dumps(dict(recovery_parent_plan_sha256=file_sha256(plan),
        selection_parent_report_sha256=file_sha256(report),case_order=[recovery.CASE])))
    done=dict(status='COMPLETE_BOUNDARY_LOWER_INDEPENDENT_RATE_SELECTION',plan_sha256=file_sha256(next_plan),
              records={**value['records'],recovery.CASE:completed()})
    next_report=folder/'report.json';next_report.write_text(json.dumps(done))
    return report,next_plan,next_report


def test_selection_accepts_complete_bound_recovery_without_replacing_parent(tmp_path):
    prior,plan,report=recovery_files(tmp_path);before=prior.read_bytes()
    assert selection.lower_selection_files(tmp_path)==(plan,report)
    assert prior.read_bytes()==before


@pytest.mark.parametrize('change',['parent','earlier_result','incomplete'])
def test_selection_rejects_stale_or_changed_or_incomplete_recovery(tmp_path,change):
    prior,plan,report=recovery_files(tmp_path)
    if change=='parent':prior.write_text(prior.read_text()+' ')
    else:
        value=json.loads(report.read_text())
        if change=='earlier_result':value['records']['fixed-two-lower-lr0']={}
        else:value['status']='STOPPED_BOUNDARY_LOWER_RATES'
        report.write_text(json.dumps(value))
    with pytest.raises(ValueError):selection.lower_selection_files(tmp_path)
