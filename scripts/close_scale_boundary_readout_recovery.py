"""Predeclared successful-TUNE diagnostic lifecycle; final FP32 weights kept."""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--case',required=True);a=p.parse_args()
    from compute_trainium.training import file_sha256
    plan=json.loads(a.plan.read_text());case=plan['cases'][a.case];sha=file_sha256(a.plan)
    code=Path(__file__).resolve().parents[1]
    if any(file_sha256(code/f)!=s for f,s in plan['source_sha256'].items()):raise AssertionError('Frozen diagnostic lifecycle source changed')
    recovery=json.loads((a.root/'scale-boundary-learning-readout-recovery-v2/plan.json').read_text())
    if file_sha256(Path(__file__))!=recovery['closure_source_sha256'] or file_sha256(a.plan)!=recovery['original_plan_sha256']:raise AssertionError('Frozen readout-only closure recovery changed')
    if time.time()>recovery['absolute_deadline_unix']:raise TimeoutError('New bounded operational closure allowance')
    if (case['phase']!='tune' or plan['storage_lifecycle']!='successful_tune_diagnostics_after_audit_score_and_exact_final_weights'):
        raise ValueError('Only the prospectively declared successful-TUNE lifecycle applies')
    audit=json.loads((a.root/case['audit_directory']/'report.json').read_text())
    score=json.loads((a.root/case['score_directory']/'report.json').read_text())
    if (audit['status']!='PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT' or audit['study_sha256']!=sha
            or score['status']!='COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT' or score['study_sha256']!=sha
            or score['retained_model_sha256']!=file_sha256(a.root/case['retained_model'])):
        raise ValueError('Complete endpoint numerical audit/readout/exact retained weights required')
    receipt=a.root/case['cleanup_receipt']
    if receipt.exists():raise FileExistsError('Preserve every successful cleanup receipt')
    native=plan['case_native_phases'][a.case]
    for identity in native:
        closure=json.loads((a.root/'snapshots'/identity/'closure.json').read_text())
        report=json.loads((a.root/'runs'/identity/'report.json').read_text())
        if closure['exit_code']!=0 or report['status']!='CAPTURED_SCALE_REGISTERED_TRAINING_PHASE':
            raise ValueError('Every exact owned Native phase must close successfully first')
    rows=[]
    for directory in [case['artifact_directory'],case['replay_directory']]:
        folder=a.root/directory
        allowed=plan.get('allowed_tune_artifact_roots',[plan['identity']+'/tune-artifacts'])
        if folder.is_symlink() or not any(folder.resolve().is_relative_to((a.root/f).resolve()) for f in allowed):
            raise ValueError('Exact private TUNE diagnostic root required')
        for file in sorted(folder.glob('*.pt')):
            if file.is_symlink():raise ValueError('No symlink diagnostic release')
            rows.append(dict(directory=directory,file=file.name,sha256=file_sha256(file),bytes=file.stat().st_size))
    if not rows:raise ValueError('Declared diagnostics unexpectedly absent')
    # Save the complete original inventory before releasing any successful
    # regenerable artifact. Raw sources, C0 and every final model stay present.
    receipt.parent.mkdir(parents=True,exist_ok=True)
    value=dict(status='INVENTORIED_SUCCESSFUL_REGENERABLE_TUNE_DIAGNOSTICS',study_sha256=sha,case=a.case,
        retained_model_sha256=score['retained_model_sha256'],original_inventory=rows,started_unix=time.time(),
        declared_before_quality=True,failures_released=False,raw_sources_released=False,final_weights_released=False)
    receipt.write_text(json.dumps(value,indent=2)+'\n')
    released=0
    for row in rows:
        file=a.root/row['directory']/row['file']
        if file_sha256(file)!=row['sha256']:raise AssertionError('Diagnostic changed after immutable inventory')
        file.unlink();released+=row['bytes']
    value.update(status='CLOSED_SUCCESSFUL_REGENERABLE_TUNE_DIAGNOSTICS',released_bytes=released,finished_unix=time.time())
    receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items() if k!='original_inventory'}),flush=True)


if __name__=='__main__':main()
