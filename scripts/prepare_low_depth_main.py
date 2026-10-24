"""Freeze the one-seed main pilot only after LR and mature-gradient gates."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root/'parcae-low-depth-pilot-v1';public=Path('results/parcae_low_depth_pilot_20261024/main')
    prep_file=out/'preparation-plan.json';prep=json.loads(prep_file.read_text())
    tune_file=out/'tuning-plan-v2.json';tune=json.loads(tune_file.read_text())
    receipts={
        'tuning':Path('results/parcae_low_depth_pilot_20261024/tuning-v2/report.json'),
        'probe_driver':Path('results/low-depth-gradient-driver-v1/report.json'),
        'selected_gradients':Path('results/low-depth-gradient-selected-v1/report.json')}
    required={'tuning':'COMPLETE_FROZEN_LOW_DEPTH_LR_SCREEN',
        'probe_driver':'PASS_LOW_DEPTH_GRADIENT_VALIDATION','selected_gradients':'PASS_LOW_DEPTH_GRADIENT_VALIDATION'}
    loaded={k:json.loads(f.read_text()) for k,f in receipts.items()}
    if any(loaded[k]['status']!=required[k] for k in loaded):raise ValueError('Main scientific launch gate unfinished')
    if len(loaded['selected_gradients']['records'])!=3:raise ValueError('All selected301M gradients must be qualified')
    selected=loaded['tuning']['selected']
    for arm in ['ordinary','fixed1','poisson2']:
        row=selected[arm]
        audit=next(r['independent_cpu'] for r in loaded['selected_gradients']['records'] if r['independent_cpu']['arm']==arm)
        if audit['checkpoint_sha256']!=row['checkpoint_sha256'] or audit['training_update']!=576:
            raise AssertionError('Selected rate and numerical checkpoint differ')
    if file_sha256(prep_file)!=tune['preparation_plan_sha256']:raise AssertionError('Fresh initialization preparation changed')
    if any(file_sha256(f)!=sha for f,sha in tune['source_sha256'].items()):raise AssertionError('Frozen tuning recipe changed')
    plan_file=out/'main-plan.json'
    if plan_file.exists():raise FileExistsError('No silent main pilot registration replacement')
    public.mkdir(parents=True,exist_ok=False)
    new_files=['scripts/prepare_low_depth_main.py','scripts/run_low_depth_main.py',
        'scripts/run_low_depth_gradient_validation.py','scripts/capture_low_depth_gradient.py',
        'scripts/audit_low_depth_gradient.py','compute_trainium/low_depth_gradient_probe.py',
        'tests/test_low_depth_gradient_probe.py','docs/LOW_DEPTH_MATURE_GRADIENT_POLICY_20261024.md',
        'docs/LOW_DEPTH_MAIN_PROTOCOL_20261024.md']
    sources={**tune['source_sha256'],**{f:file_sha256(f) for f in new_files}}
    cases={};identities=[];warm_seconds=0;extra_branch_work=0
    for arm in ['ordinary','fixed1','poisson2']:
        row=next(r for r in prep['records'] if r['arm']==arm);endpoint=row['matched_work'];equal=row['equal_data']
        work=np.load(out/row['work_table'],mmap_mode='r');main=arm+'-main-s1231';short=arm+'-equal-s1231'
        common={'arm':arm,'seed':1231,'state_seed':3240,'parameters':row['parameters'],
            'initial_state':'s1231-'+('ordinary' if arm=='ordinary' else 'family'),
            'peak_lr':selected[arm]['peak_lr'],'warmup_targets':50_000_000,
            'depth_table':row['depth_table'],'depth_table_sha256':row['depth_table_sha256'],
            'work_table':row['work_table'],'work_table_sha256':row['work_table_sha256'],
            'primary_inference_depth':2 if arm=='poisson2' else 1}
        cases[main]={**common,'cooldown_start_update':endpoint['cooldown_start_update'],
            'cooldown_start_targets':endpoint['cooldown_start_targets'],
            'total_updates':endpoint['updates'],'end_targets':endpoint['real_targets'],
            'checkpoint_updates':sorted({576,960,1920,endpoint['cooldown_start_update'],endpoint['updates']}),
            'segment_ends':[1920,endpoint['updates']],'training_matmuls':endpoint['executed_matmuls']}
        cases[short]={**common,'cooldown_start_update':960,'cooldown_start_targets':equal['cooldown_start_targets'],
            'total_updates':1920,'end_targets':equal['real_targets'],'checkpoint_updates':[1920],
            'segment_ends':[1920],'training_matmuls':int(work[:1920].sum()),'parent':{'case_id':main,'update':960}}
        if endpoint['cooldown_start_update']<=1920:raise ValueError('Main1B gradient gate must precede main decay')
        identities.extend(f'low-depth-main-v1-{arm}-{phase}' for phase in ['first','finish','equal'])
        branch_targets=equal['real_targets']-equal['cooldown_start_targets']
        warm_seconds+=(endpoint['real_targets']+branch_targets)/tune['measured_eight_chip_targets_per_second'][arm]
        extra_branch_work+=int(work[960:1920].sum())
    plan={'identity':'parcae-low-depth-main-v1','registered_unix':time.time(),
        'purpose':'natural_low_depth_scientific_pilot','launch_gates_complete':True,
        'source_sha256':sources,'worker_source_sha256':{f:sha for f,sha in sources.items() if f.startswith(('scripts/','compute_trainium/','configs/'))},
        'model':prep['model'],'initial_states':prep['initial_states'],'precision':'fp16','accumulation':16,
        'optimizer_scalar_input_policy':tune['optimizer_scalar_input_policy'],
        'routing_seed':271828,'cases':cases,'data':tune['data'],'evaluation':tune['evaluation'],
        'root_comm_ports':reserve_ports(a.root,identities),
        'selected_rates':{arm:selected[arm]['peak_lr'] for arm in selected},
        'launch_gate_receipts':{k:{'path':str(f),'sha256':file_sha256(f)} for k,f in receipts.items()},
        'tuning_plan_sha256':file_sha256(tune_file),'preparation_plan_sha256':file_sha256(prep_file),
        'matched_main_budget_per_arm':prep['reference_budget_matmuls'],
        'extra_equal_data_branch_matmuls':extra_branch_work,
        'estimated_native_warm_reservation_seconds':warm_seconds/2,
        'measured_eight_chip_targets_per_second':tune['measured_eight_chip_targets_per_second'],
        'seconds_limit':8*3600,'native_phase_seconds_limit':12000,'cpu_score_seconds_limit':1200,
        'numerical_checkpoints':['stable1B','completed_matched_work','completed_equal_data'],
        'primary_contrast':'poisson2 versus true one-pass Parcae at matched complete estimated training matrix work',
        'secondary_contrasts':['candidate versus recipe-matched ordinary at matched work',
            'all arms at independently annealed equal data','same candidate checkpoint at supported exits1,2,3'],
        'scope':'Single fresh paired seed1231 on consumedDEV,not terminal confirmation. Fresh random weights,declared matched complete matrix budget,own cooldowns,extra equal-data branches charged. Three selected rates pass mature-gradient checks; each main1B/endpoint is independently checked too. Strongest cheap-connection confirmation remains necessary. No fresh TEST,alpha,teacher,RL,synthetic corpus or adaptive route.'}
    write_state(plan_file,plan);write_state(public/'plan.json',plan)
    print(json.dumps({'identity':plan['identity'],'warm_reservation_hours':warm_seconds/7200,
                      'hard_execution_hours':8,'matched_budget':prep['reference_budget_matmuls']}))


if __name__=='__main__':main()
