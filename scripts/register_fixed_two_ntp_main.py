"""Admit the finite six-endpoint curve only after the bounded TUNE passes."""
import copy
import json
import math
from pathlib import Path
import shutil
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.fixed_two_curve import curve_nodes,validate_curve
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='fixed-two-ntp-main-v1'
PUBLIC=Path('results/fixed_two_small_core_main_20261025')


def register(root):
    from compute_trainium.inode_reserve import require_inode_reserve
    require_inode_reserve([root,Path('/')])
    if shutil.disk_usage(root).free<70*1024**3 or shutil.disk_usage('/').free<35*1024**3:raise RuntimeError('MAIN needs its projected state space in addition to the protected35GiB')
    if managed_running():raise RuntimeError('Native predecessor owners must be terminal')
    tune_dir=root/'fixed-two-ntp-tune-v1';tune_file=tune_dir/'report.json';tune=json.loads(tune_file.read_text());old=json.loads((tune_dir/'plan.json').read_text())
    if tune['status']!='COMPLETE_FIXED_TWO_NTP_TUNING' or tune['plan_sha256']!=file_sha256(tune_dir/'plan.json'):raise ValueError('Complete frozen TUNE required')
    if len(tune['prefix_gates'])!=3 or any(x['independent_cpu']['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT' for x in tune['prefix_gates'].values()) or tune['selected_final_gate']['independent_cpu']['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT':raise ValueError('All counted prefix and selected endpoint gradient gates required')
    winner=min(tune['scores'],key=lambda key:(tune['scores'][key]['metrics']['weighted_nll'],old['cases'][key]['peak_lr']))
    if set(tune['scores'])!=set(old['cases']) or winner!=tune['selected_case'] or old['cases'][winner]['peak_lr']!=tune['selected_rate'] or tune['selected_final_gate']['independent_cpu']['case_id']!=winner:raise AssertionError('Declared complete-grid selection differs')
    prep=root/'fixed-two-ntp-main-preparation-v1';ready=json.loads((prep/'report.json').read_text());manifest_file=prep/'manifest.json';m=json.loads(manifest_file.read_text());registration=json.loads((prep/'registration.json').read_text())
    if ready['status']!='COMPLETE_FIXED_TWO_NTP_MAIN_PREPARATION' or file_sha256(manifest_file)!=ready['manifest_sha256'] or not m['original_anchor_prefix_exact']:raise ValueError('Verified complete source/counter preparation required')
    if any(file in old['source_sha256'] and old['source_sha256'][file]!=sha for file,sha in registration['source_sha256'].items()):raise AssertionError('Preparation conflicts with the qualified scientific stack')
    for file,sha in {**old['source_sha256'],**registration['source_sha256']}.items():
        if file_sha256(file)!=sha:raise AssertionError('Qualified TUNE/preparation source changed')
    if m['model']!=old['model'] or m['future_config']!=old['future_config'] or m['data']!=old['data']:raise AssertionError('MAIN changes the qualified model or original source')
    one_rate=tune['historical_one_selection']['peak_lr'];fixed_rate=tune['selected_rate'];cases={}
    for key,e in m['endpoints'].items():
        arm='one' if e['arm']=='one' else 'recurrent';table=m['tables'][e['arm']]
        cases[key]=dict(cell=arm+'-ntp',arm=arm,objective_weight=0.,seed=5231,state_seed=7240,initial_state='s5231-family',training_parameters=m['initial_state']['training_parameters'],peak_lr=one_rate if arm=='one' else fixed_rate,warmup_targets=50000000,cooldown_start_update=e['cooldown_start_update'],cooldown_start_targets=e['cooldown_start_targets'],total_updates=e['total_updates'],end_targets=e['end_targets'],end_anchors=e['end_anchors'],training_dense_matrix_ops=e['training_matrix_ops'],depth_table='../'+prep.name+'/'+table['depth_file'],depth_table_sha256=table['depth_sha256'],work_table='../'+prep.name+'/'+table['work_file'],work_table_sha256=table['work_sha256'],checkpoint_updates=[960,1920,3840] if key in ['one-2b','fixed-2b'] else [e['total_updates']],segment_ends=[960,1920,3840] if key in ['one-2b','fixed-2b'] else [e['total_updates']],gradient_updates=[e['total_updates']],primary_inference_depth=1 if arm=='one' else 2)
    for key,source,update in [('one-1b','one-2b',960),('fixed-1b','fixed-2b',960),('one-work-1b','one-2b',960),('one-work-2b','one-2b',1920)]:cases[key]['parent']=dict(case_id=source,update=update)
    nodes=curve_nodes(cases);actual_updates=validate_curve(cases,nodes)
    hardware_file=root/'future-span-math-qualification-v2/report.json';hardware=json.loads(hardware_file.read_text())
    if file_sha256(hardware_file)!=old['launch_receipts']['future-span-math-qualification-v2']['report_sha256']:raise AssertionError('Original One timing changed')
    warm=next(r['native']['warm'] for r in hardware['records'] if r['case']['identity']=='one-ntp')
    one_seconds=warm['warm_seconds']/100;fixed_seconds=max(v['warm_seconds']/100 for v in tune['warm'].values())
    if any(v['warm_updates']!=100 for v in tune['warm'].values()) or not all(math.isfinite(v) and v>0 for v in [one_seconds,fixed_seconds]):raise AssertionError('Actual100-update timings required')
    duration=sum((n['end']-(nodes[n['parent']]['end'] if n['parent'] else 0))*(one_seconds if cases[n['case']]['arm']=='one' else fixed_seconds) for n in nodes.values())
    cold=max(old['measured_budget']['cold_allowance'],1.5*(warm['wall_seconds']-warm['warm_seconds']))
    estimate=duration*1.5/2+(len(nodes)+len(cases))*cold/2+len(cases)*900/3+(len(cases)+1)*360/3+300
    if estimate>18000:raise RuntimeError('Complete fixed curve does not fit the proposed measured five-hour cap')
    extras=['scripts/register_fixed_two_ntp_main.py','scripts/run_fixed_two_ntp_main.py','compute_trainium/fixed_two_curve.py','tests/test_fixed_two_curve.py','docs/FIXED_TWO_SMALL_CORE_MAIN_ADMISSION_20261025.md','scripts/prepare_fixed_two_ntp_main.py','scripts/evaluate_fixed_two_curve_cpu.py','compute_trainium/reference_loss_strata.py','tests/test_reference_loss_strata.py','docs/FIXED_TWO_NTP_STRATA_20261025.md']
    reference_file=root/'future-span-work-control-v1/plan.json';reference_plan=json.loads(reference_file.read_text());reference_key='one-ntp-s7283-lr32000-work'
    reference_case=reference_plan['cases'][reference_key];reference_checkpoint=root/'runs'/('future-span-work-control-v1-'+reference_key+'-finish')/'checkpoint-u2365.pt';reference_fit=json.loads((reference_checkpoint.parent/'report.json').read_text());reference_state=next(v for v in reference_fit['checkpoint_records'] if v['update']==2365)
    if reference_case['seed']!=7283 or reference_case['objective_weight']!=0. or reference_plan['model']!=m['model'] or file_sha256(reference_checkpoint)!=reference_state['checkpoint_sha256'] or not reference_state['all_complete_replica_states_equal']:raise AssertionError('Independent frozen difficulty reference differs')
    reference=dict(plan=str(reference_file.relative_to(root)),plan_sha256=file_sha256(reference_file),case=reference_key,checkpoint=str(reference_checkpoint.relative_to(root)),checkpoint_sha256=reference_state['checkpoint_sha256'],depth=1,seed=7283,scope='Independent-seed existing One/NTP work checkpoint;difficulty stratification only,no new fit or primary comparator')
    parity_file=root/'fixed-two-strata-reference-v1/report.json';parity=json.loads(parity_file.read_text());reference_score=root/'fixed-two-strata-reference-v1/cpu/report.json'
    if parity['status']!='PASS_POSITION_LOSS_READER_PARITY' or parity['difference_from_original']!=0. or file_sha256(reference_score)!=parity['score_report_sha256']:raise AssertionError('Unchanged full-panel scalar score and position-reader parity required')
    reference.update(prepared_directory='fixed-two-strata-reference-v1/cpu',prepared_score_sha256=parity['score_report_sha256'],parity_report_sha256=file_sha256(parity_file))
    sources={**old['source_sha256'],**registration['source_sha256'],**{f:file_sha256(f) for f in extras}};worker,host=split_snapshot_sources(sources)
    names=[IDENTITY+'-'+name for name in nodes]+[IDENTITY+'-'+key+'-gradient' for key in cases]
    plan=copy.deepcopy(old)
    for stale in ['historical_one_tuning','historical_one_selection','predecessor_report_sha256','primary_scope','selected_rates']:plan.pop(stale,None)
    plan.update(identity=IDENTITY,registered_unix=time.time(),stage='main_fixed_two_curve',cases=cases,nodes=nodes,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,initial_states={'s5231-family':m['initial_state']},auxiliary_data=m['auxiliary_data'],root_comm_ports=reserve_ports(root,names),seconds_limit=18000,phase_seconds_limit=4800,cpu_phase_seconds_limit=1200,tune_plan_sha256=file_sha256(tune_dir/'plan.json'),tune_report_sha256=file_sha256(tune_file),preparation_manifest_sha256=file_sha256(manifest_file),rate_selection=dict(one=one_rate,fixed=fixed_rate,source='Separate declared three-rate cooled301M TUNE;edge selections remain unbracketed'),work_matches=m['work_matches'],minimum_useful_gain_nats=.01,primary_comparison='one-work-2b minus fixed-2b weighted_nll',primary_scope='One paired consumed-development seed;conditional source-group intervals are not training-seed confirmation',planned_actual_research_updates=actual_updates,measured_budget=dict(one_warm_seconds_per_update=one_seconds,fixed_warm_seconds_per_update=fixed_seconds,cold_allowance=cold,predicted_seconds_with_margin=estimate,cap_seconds=18000),scope=Path(extras[4]).read_text())
    plan.update(stratification_reference=reference,secondary_diagnostic_scope=Path('docs/FIXED_TWO_NTP_STRATA_20261025.md').read_text(),learning_rate_schedule=dict(kind='qualified_absolute_WSD_cosine',warmup_targets=50000000,cooldown_fraction=.5,final_lr=0.,final_muon_cautious_decay=0.,description_correction='Supersedes the unadmitted preparation draft10% wording;executed qualified schedule unchanged'))
    out=root/IDENTITY;out.mkdir();PUBLIC.mkdir();write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    return plan
