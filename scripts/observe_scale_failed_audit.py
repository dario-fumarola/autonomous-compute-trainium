"""Capture missing failure metrics with the unchanged frozen audit calculation."""
import argparse
import json
from pathlib import Path
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def observe(root, frozen_code, plan, case, folder):
    if folder.exists():raise FileExistsError('Preserve every failed-audit observation')
    original = frozen_code / 'scripts/audit_scale_registered.py'
    registration = json.loads(plan.read_text())
    if file_sha256(original) != registration['source_sha256']['scripts/audit_scale_registered.py']:
        raise AssertionError('Original frozen audit source changed')
    source = original.read_text()
    marker = "    if (not delta['passes_two_percent_gate'] or abs(difference)>.003 or actual['signature']!=signature\n"
    if source.count(marker) != 1:raise AssertionError('One exact original diagnostic insertion point required')
    added = """    (output/'full-gradient-observation.json').write_text(json.dumps(dict(
        gradient=delta,native_minus_cpu_nll=difference,cpu_nll=total/targets,
        cpu_norm_before_clip=norm,windows=512,source_cursor=cursor,real_targets=targets,
        signature_matches=actual['signature']==signature,
        source_cursor_matches=actual['source_cursor']==cursor,
        real_targets_match=actual['real_targets']==targets,native_clipped=actual['clipped'],
        model_unchanged=digest(dict(model.state_dict()))==model_digest),indent=2)+'\\n')
"""
    patched = source.replace(marker, added + marker)
    namespace = dict(__name__='registered_read_only_failure_observer', __file__=str(original))
    exec(compile(patched, str(original), 'exec'), namespace)
    folder.mkdir(); started = time.time()
    scope = dict(registered_unix=started, original_plan_sha256=file_sha256(plan),
                 original_audit_sha256=file_sha256(original), observer_source_sha256=file_sha256(Path(__file__)),
                 case=case, research_optimizer_updates=0, assigned_chips=0,
                 maximum_seconds=1800,
                 scope='Repeat only the exact frozen independent512 calculation at the unchanged saved checkpoint/batch/noise. Insert observation serialization before its original gate. No changed arithmetic,threshold,Native execution,training,quality readout or retroactive certification.')
    write_state(folder/'plan.json',scope)
    result=dict(status='STOPPED_READ_ONLY_AUDIT_OBSERVATION',assigned_chips=0,research_optimizer_updates=0)
    try:
        namespace['audit'](root,plan,case,folder/'audit')
        result['status']='ORIGINAL_AUDIT_REPLAY_PASSED_UNEXPECTEDLY_NOT_RETROACTIVE_CERTIFICATION'
    except AssertionError as error:
        if str(error)!='Independent complete512 gradient/NLL failed':raise
        if not (folder/'audit/full-gradient-observation.json').exists():raise
        result.update(status='OBSERVED_ORIGINAL_FULL512_NUMERICAL_REJECTION',error=str(error))
    finally:
        result['wall_seconds']=time.time()-started
        result['original_stopped_history_unchanged']=True
        write_state(folder/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--frozen-code',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();observe(a.root,a.frozen_code,a.plan,a.case,a.output)
