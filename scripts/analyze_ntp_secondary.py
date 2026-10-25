"""Read already frozen MAIN outputs for a TUNE-selected NTP secondary candidate."""
import argparse
import json
import os
import time
from pathlib import Path

from compute_trainium.metrics import DocumentScore,paired_group_interval
from compute_trainium.ntp_secondary import ntp_point_contrasts,discovery_decision
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='ntp-secondary-discovery-v1'
PUBLIC=Path('results/ntp_secondary_discovery_20261025')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();out=a.root/IDENTITY
    plan_file=out/'plan.json';main_root=a.root/'future-span-main-v2';tune_root=a.root/'future-span-tune-v2'
    if not plan_file.exists():
        if (main_root/'plan.json').exists():raise ValueError('Register before MAIN admission/outcomes')
        source=['scripts/analyze_ntp_secondary.py','compute_trainium/ntp_secondary.py',
                'compute_trainium/metrics.py','tests/test_ntp_secondary.py']
        execution=json.loads((tune_root/'execution/execution-start.json').read_text())
        tuning=json.loads(Path('results/future_span_tune_20261025/primitive-v2/scores-progress.json').read_text())
        observed=[r for r in tuning['records'] if r['cell']=='recurrent-ntp']
        if len(observed)!=3:raise ValueError('All three NTP recurrent tuning opportunities required to motivate this secondary')
        plan={'identity':IDENTITY,'registered_unix':time.time(),'selected_after_observing_tune':True,
            'source_sha256':{f:file_sha256(f) for f in source},'main_identity':'future-span-main-v2',
            'tune_plan_sha256':file_sha256(tune_root/'plan.json'),'observed_tune_case_scores':[
                {'case_id':r['case_id'],'nll':r['metrics']['weighted_nll']} for r in observed],
            'seeds':[5231,7283],'primary_inference_depth':2,'useful_gain_nats':.01,
            'contrasts':['NTP recurrence over one-pass NTP','NTP recurrence over one-pass future loss',
                         'same-checkpoint supported recurrent NTP depth1-to2'],
            'bootstrap_draws':2000,'bootstrap_seed':81251,
            'latest_parent_finish_unix':execution['deadline_unix']+8*3600,
            'cpu_seconds_limit':180,'terminal_alpha_spent':0,'test_accessed':False,
            'scope':'Secondary discovery selected after NTP TUNE gain and registered before MAIN admission/outcomes. Original four-cell primary future-objective decision remains unchanged and reported even if negative. Reuses all existing fixed MAIN readouts;no new training,forward,checkpoint selection or data. Both pilot seeds must clear all three point gates;strongest ordinary/cheap baselines,matched-work controls and >=5 fresh paired confirmation seeds/source groups with allocated alpha are still required.'}
        out.mkdir(exist_ok=False);PUBLIC.mkdir(parents=True,exist_ok=False)
        write_state(plan_file,plan);write_state(PUBLIC/'plan.json',plan)
    else:plan=json.loads(plan_file.read_text())
    if a.prepare_only:return
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal secondary readout')
    while True:
        if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Frozen secondary analysis changed')
        report_file=main_root/'report.json'
        if report_file.exists():
            parent=json.loads(report_file.read_text())
            if parent['status']!='COMPLETE_BOUNDED_FUTURE_SPAN_OBJECTIVE_PILOT':
                raise RuntimeError('Complete MAIN primary results required;do not select partial favorable cells')
            break
        for file in [tune_root/'report.json',Path('results/future_span_science_queue_20261025/primitive-v2/report.json')]:
            if file.exists() and json.loads(file.read_text())['status'].startswith('STOPPED'):
                raise RuntimeError('Parent stopped;no secondary claim')
        if time.time()>plan['latest_parent_finish_unix']:raise TimeoutError('Original bounded parent window expired')
        write_state(PUBLIC/'status.json',{'status':'WAITING_FOR_COMPLETE_UNCHANGED_MAIN_STUDY','assigned_chips':0})
        time.sleep(10)
    started=time.time();points={};intervals={};inputs=[]
    final=[r for r in parent['scores'] if r['readout']=='cooled1b']
    if len(final)!=16:raise ValueError('All sixteen declared final paths required')
    gates=parent['independent_gradient_gates']
    if len(gates)!=8 or len({r['case_id'] for r in gates})!=8:raise ValueError('All eight unique final gradient gates required')
    for seed in plan['seeds']:
        rows=[r for r in final if r['seed']==seed];points[seed]=ntp_point_contrasts(rows);docs={}
        for r in rows:
            if (r['cell'],r['depth']) not in {('one-ntp',1),('one-future',1),('recurrent-ntp',1),('recurrent-ntp',2)}:continue
            gate=next(g for g in gates if g['case_id']==r['case_id'])['independent_cpu']
            if (gate['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
                    or gate['checkpoint_sha256']!=r['checkpoint_sha256']):raise ValueError('Exact qualified endpoint required')
            file=main_root/'scores'/f'{r["case_id"]}-cooled1b-d{r["depth"]}'/'document-scores.json'
            inputs.append({'case_id':r['case_id'],'depth':r['depth'],'sha256':file_sha256(file)})
            docs[r['cell'],r['depth']]=[DocumentScore(v['domain'],v['id'],v['group'],v['nll_sum'],v['real_targets'])
                for v in json.loads(file.read_text())['records']]
        target=docs['recurrent-ntp',2]
        intervals[seed]={name:paired_group_interval(control,target,draws=plan['bootstrap_draws'],
            seed=plan['bootstrap_seed']+seed) for name,control in [
                ('over_one_ntp',docs['one-ntp',1]),('over_one_future',docs['one-future',1]),
                ('supported_depth1_to2',docs['recurrent-ntp',1])]}
        if time.time()-started>plan['cpu_seconds_limit']:raise TimeoutError('Fixed secondary CPU cap')
    result={'status':'COMPLETE_NTP_SECONDARY_DISCOVERY_READOUT','points_by_pilot_seed':points,
        'conditional_source_intervals':intervals,'secondary_decision':discovery_decision(points),
        'unchanged_primary_future_objective_decision':parent['decision'],
        'parent_plan_sha256':parent['plan_sha256'],'parent_report_sha256':file_sha256(report_file),
        'input_document_score_hashes':inputs,'registration_sha256':file_sha256(plan_file),
        'wall_seconds':time.time()-started,'assigned_chips':0,'optimizer_updates':0,
        'extra_model_forwards':0,'test_accessed':False,'terminal_alpha_spent':0,'terminal_success':False,
        'scope':plan['scope']}
    write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)
    write_state(PUBLIC/'status.json',{'status':result['status'],'assigned_chips':0})


if __name__=='__main__':
    try:main()
    except BaseException as error:
        parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--root',type=Path)
        parsed,_=parser.parse_known_args()
        if parsed.root and (parsed.root/IDENTITY/'plan.json').exists() and not (parsed.root/IDENTITY/'report.json').exists():
            stopped={'status':'STOPPED_NTP_SECONDARY_DISCOVERY_READOUT','error':str(error),
                'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,'terminal_success':False}
            write_state(parsed.root/IDENTITY/'report.json',stopped);write_state(PUBLIC/'report.json',stopped)
            write_state(PUBLIC/'status.json',{'status':stopped['status'],'assigned_chips':0})
        raise
