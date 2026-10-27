"""Unchanged frozen audit math/gates, with metrics saved before rejection."""
import argparse
import json
from pathlib import Path


def instrument(original):
    source=original.read_text()
    marker="    if (not delta['passes_two_percent_gate'] or abs(difference)>.003 or actual['signature']!=signature\n"
    if source.count(marker)!=1:raise AssertionError('One exact unchanged audit gate required')
    added="""    (output/'full-gradient-observation.json').write_text(json.dumps(dict(
        gradient=delta,native_minus_cpu_nll=difference,cpu_nll=total/targets,
        cpu_norm_before_clip=norm,windows=512,source_cursor=cursor,real_targets=targets,
        signature_matches=actual['signature']==signature,
        source_cursor_matches=actual['source_cursor']==cursor,
        real_targets_match=actual['real_targets']==targets,native_clipped=actual['clipped'],
        model_unchanged=digest(dict(model.state_dict()))==model_digest),indent=2)+'\\n')
"""
    namespace=dict(__name__='registered_audit_with_failure_observations',__file__=str(original))
    exec(compile(source.replace(marker,added+marker),str(original),'exec'),namespace)
    return namespace['audit']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--case',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:instrument(Path(__file__).with_name('audit_scale_registered.py'))(a.root,a.plan,a.case,a.output)
    except BaseException as error:
        if a.output.exists():
            (a.output/'failure.json').write_text(json.dumps(dict(status='STOPPED_SCALE_TRAINING_AUDIT',
                error_type=type(error).__name__,error=str(error)))+'\n')
        raise
