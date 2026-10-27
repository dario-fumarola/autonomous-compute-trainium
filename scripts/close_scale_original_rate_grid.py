"""Zero-update certificate for every fixed opportunity, preserving two rejects."""
import argparse
import json
import math
from pathlib import Path
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_scale_last_original_rates import rejection
from scripts.run_scale_boundary_lower_rates import PARENT,COMPLETION,ORIGINAL_CERTIFICATE,select

PUBLIC=Path('results/scale_original_rate_closure_20261027')


def valid_rejected_observation(value):
    return (value['windows']==512 and not value['gradient']['passes_two_percent_gate']
        and math.isfinite(value['native_minus_cpu_nll']) and abs(value['native_minus_cpu_nll'])<=.003
        and all(value[k] for k in ['signature_matches','source_cursor_matches','real_targets_match',
                                   'native_clipped','model_unchanged']))


def close(root):
    if managed_running():raise RuntimeError('Every Native worker must be closed')
    folder=root/ORIGINAL_CERTIFICATE
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every explicit certificate')
    base_plan=root/PARENT/'plan.json';base_report=root/PARENT/'report.json'
    last_plan=root/COMPLETION/'plan.json';last_report=root/COMPLETION/'report.json'
    plan=json.loads(last_plan.read_text());result=json.loads(last_report.read_text())
    if (result['status']!='STOPPED_LAST_ORIGINAL_RATE_COMPLETION'
            or result['error']!='CPU phase failed: fixed-two-lr3-audit'
            or result['plan_sha256']!=file_sha256(last_plan)
            or plan['original_stopped_parent_plan_sha256']!=file_sha256(base_plan)
            or plan['parent_report_sha256']!=file_sha256(base_report)):
        raise ValueError('Exact stopped last-stage and original parent required')
    records=dict(result['records']);old_rejection=rejection(root)
    if any(records['fixed-two-lr2'][k]!=v for k,v in old_rejection.items()):
        raise AssertionError('First actual numerical rejection changed')
    observation=root/COMPLETION/'audits/fixed-two-lr3/full-gradient-observation.json'
    failure=root/COMPLETION/'audits/fixed-two-lr3/failure.json'
    metrics=json.loads(observation.read_text())
    if (not valid_rejected_observation(metrics)
            or json.loads(failure.read_text())['error']!='Independent complete512 gradient/NLL failed'
            or (root/COMPLETION/'scores/fixed-two-lr3/report.json').exists()):
        raise ValueError('Actual full512 numerical rejection without quality access required')
    record=records['fixed-two-lr3']
    for phase in ['warm','replay']:
        if (record[phase]['status']!='CAPTURED_SCALE_REGISTERED_TRAINING_PHASE'
                or record[phase]['stop_update']!=300 or not record[phase]['cooldown_complete']
                or record[phase]['plan_sha256']!=file_sha256(last_plan)):
            raise AssertionError('Both original closed terminal captures required')
    for identity in plan['case_native_phases']['fixed-two-lr3']:
        if json.loads((root/'snapshots'/identity/'closure.json').read_text())['exit_code']!=0:
            raise AssertionError('Exact Native closure failed')
    records['fixed-two-lr3']={**record,'declared_numerically_ineligible':True,
        'reason':'Original mature512 numerical threshold failed; no admissible quality score',
        'original_audit_failure_sha256':file_sha256(failure),
        'repeated_observation_sha256':file_sha256(observation)}
    original=json.loads((root/'scale-medium-tune-65536-v1/plan.json').read_text())
    base=json.loads(base_plan.read_text());verified=[]
    for key in original['case_order']:
        item=records[key]
        if item.get('declared_numerically_ineligible'):
            if 'score' in item:raise ValueError('Rejected history must not have a prediction score')
            continue
        if 'source_plan' in item:source=root/item['source_plan']
        elif key in plan['cases']:source=last_plan
        else:source=base_plan
        origin=json.loads(source.read_text());case=origin['cases'][key];audit=item['audit'];score=item['score']
        if (audit['study_sha256']!=file_sha256(source) or score['study_sha256']!=file_sha256(source)
                or audit['status']!='PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT'
                or audit['full512']['windows']!=512 or not audit['full512']['gradient']['passes_two_percent_gate']
                or abs(audit['full512']['native_minus_cpu_nll'])>.003 or not audit['replay']['passes']
                or score['status']!='COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT'
                or score['documents']!=192 or not score['model_unchanged']
                or score['real_targets']!=157013162 or case['endpoint_updates']!=300
                or case['peak_lr']!=original['cases'][key]['peak_lr']
                or file_sha256(root/case['initializer'])!=case['initializer_sha256']
                or file_sha256(root/case['retained_model'])!=score['retained_model_sha256']):
            raise ValueError('Every admitted original opportunity must retain complete exact proof: '+key)
        for identity in origin['case_native_phases'][key]:
            if json.loads((root/'snapshots'/identity/'closure.json').read_text())['exit_code']!=0:
                raise AssertionError('Admitted original Native closure failed')
        verified.append(key)
    if len(verified)!=6:raise AssertionError('Exactly six qualified and two rejected opportunities expected')
    rates={k:original['cases'][k]['peak_lr'] for k in original['case_order']}
    selected=select(records,rates,expected=4)
    certificate={**plan,'identity':ORIGINAL_CERTIFICATE,'registered_unix':time.time(),
        'closure_source_sha256':file_sha256(Path(__file__)),
        'completed_last_stage_plan_sha256':file_sha256(last_plan),
        'completed_last_stage_report_sha256':file_sha256(last_report),
        'declared_ineligible_records':{k:v for k,v in records.items() if v.get('declared_numerically_ineligible')},
        'new_native_executions':0,'new_optimizer_updates':0,
        'scope':'Read-only explicit completion of eight original opportunities: six fully qualified retained prediction models,two actual mature512 numerical rejections with no quality score. Both stopped predecessor reports remain stopped. No retry,threshold change,precision change or main admission.'}
    folder.mkdir();PUBLIC.mkdir();write_state(folder/'plan.json',certificate)
    value=dict(status='COMPLETE_BOUNDARY_FIXED_GRID_WITH_NUMERICAL_REJECTION',
        plan_sha256=file_sha256(folder/'plan.json'),records=records,selected=selected,
        qualified_original_cases=verified,numerically_ineligible_cases=['fixed-two-lr2','fixed-two-lr3'],
        new_optimizer_updates=0,assigned_chips=0,scientific_main_fit_admitted=False,
        stopped_predecessors_unchanged=True)
    write_state(folder/'report.json',value)
    write_state(PUBLIC/'plan.json',certificate)
    write_state(PUBLIC/'summary.json',{k:v for k,v in value.items() if k!='records'})
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    value=close(a.root);print(json.dumps({k:v for k,v in value.items() if k!='records'}))
