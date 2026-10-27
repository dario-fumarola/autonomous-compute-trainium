"""Fresh scalar-corrected TUNE, preserving the rejected original history."""
import argparse
import json
from pathlib import Path

from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts import run_scale_tune as engine

IDENTITY='scale-medium-tune-65536-v1'
PUBLIC=Path('results/scale_medium_tune_65536_20261027')


def prepare(root):
    numerical={}
    for label in ['fixed-two','ordinary']:
        identity='scale-65536-'+label+'-numerical-v1';file=root/identity/'completion-verification.json'
        result=json.loads(file.read_text())
        if result['status']!='PASS_SCALE_65536_COMPLETE_NUMERICAL_QUALIFICATION':raise ValueError('Both complete higher-scale numerical qualifications required')
        numerical[identity]=file_sha256(file)
    original=root/'scale-medium-tune-v1/report.json'
    if json.loads(original.read_text())['status']!='STOPPED_SCALE_TUNE':raise ValueError('Original rejected study stays preserved')
    if list((root/'scale-medium-tune-v1/scores').glob('*/report.json')):raise ValueError('No original readout may be used in this prospective correction')
    engine.prepare(root,tune_seconds_limit=13500,main_required_seconds=14*3600,nvme_allowance_gib=13)
    file=root/IDENTITY/'plan.json';plan=json.loads(file.read_text())
    plan.update(manual_loss_scale=65536,numerical_gates=numerical,
        original_rejected_tune_report_sha256=file_sha256(original),
        allowed_tune_artifact_roots=[IDENTITY+'/tune-artifacts','../recurrence/'+IDENTITY+'/tune-artifacts'],
        scope=plan['scope']+'\nFresh corrected numerical recipe:manual loss scaler65536 in both arms, identical FP16/FP32 operators/objective/architecture. The original157M rejected trial stays ineligible and charged. Both newly qualified100-warm/first-update/full512/replay gates precede all quality readouts. Original unupdated38213 C0 is reused; no trained or qualification-updated weights enter TUNE. Every role receives the same four300-update trials and completed cooldowns. Independent first-update math is already qualified,so TUNE retains only its mature299/300 states,full512 mature gradient and actual replay,with first64 consumed-DEV tune documents per domain and the causal window-ordinal noise policy.3.75-hour total TUNE cap plus14-hour measured main allowance stays inside the unchanged original24-hour cap. All outcomes and failures remain. No main or confirmation automatically admitted.')
    for key in plan['case_order']:
        case=plan['cases'][key]
        case.update(audit_first_update=False,save_updates=[299,300],capture_before_updates=[299])
        replay='../recurrence/'+IDENTITY+'/tune-artifacts/'+key+'/replay'
        case['replay_directory']=replay
        plan['cases'][key+'-replay'].update(artifact_directory=replay,audit_first_update=False,
            replay_directory=replay,save_updates=[300],capture_before_updates=[])
    write_state(file,plan);write_state(PUBLIC/'plan.json',plan)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    engine.IDENTITY=IDENTITY;engine.PUBLIC=PUBLIC
    engine.run(a.root) if a.run else prepare(a.root)
