"""Register the bounded four-LR/three-arm screen after all driver gates pass."""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


RATES=[.002,.004,.008,.016]
ARMS=['ordinary','fixed1','poisson2']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root/'parcae-low-depth-pilot-v1';public=Path('results/parcae_low_depth_pilot_20261024/tuning-v2')
    public.mkdir(parents=True,exist_ok=True)
    preparation=json.loads((out/'preparation-plan.json').read_text())
    if json.loads((out/'preparation-report.json').read_text())['status']!='COMPLETE_LOW_DEPTH_SCIENTIFIC_PREPARATION':
        raise ValueError('Fresh scientific initialization/data/work preparation required')
    paths={'model_training':Path('results/parcae_runtime_scalars_20261024/report.json'),
           'replayed_gradient':Path('results/parcae_runtime_scalars_20261024/replayed-gradient-report.json'),
           'driver':Path('results/low_depth_driver_qualification_20261024/attempt2/report.json')}
    statuses={'model_training':'PASS_COMPLETE_PARCAE_RUNTIME_SCALAR_QUALIFICATION',
              'replayed_gradient':'PASS_RUNTIME_REPLAYED_INDEPENDENT_GRADIENTS',
              'driver':'PASS_FINITE_LOW_DEPTH_DRIVER_AND_BRANCH_RECOVERY'}
    for name,path in paths.items():
        if json.loads(path.read_text())['status']!=statuses[name]:raise ValueError('Unfinished scientific launch gate: '+name)
    plan_file=out/'tuning-plan-v2.json'
    if plan_file.exists():raise FileExistsError('No silent scientific tuning registration replacement')
    files=['scripts/prepare_low_depth_tuning_v2.py','scripts/run_low_depth_tuning_v2.py',
        'scripts/train_low_depth_runtime.py','compute_trainium/parcae_runtime_scalars.py',
        'scripts/audit_runtime_replayed_gradients.py',
        'docs/LOW_DEPTH_TUNING_RUNTIME_AMENDMENT_20261024.md','tests/test_parcae_runtime_scalars.py',
        'scripts/train_low_depth_scientific.py','scripts/evaluate_low_depth_cpu.py',
        'scripts/train_parcae_precise_qualification.py','scripts/qualify_native.py',
        'scripts/qualify_parcae_candidate_cpu.py','scripts/qualify_parcae_optimizer_cpu.py',
        'scripts/audit_parcae_precise_training_point.py','third_party/parcae_reference/muon_adamw.py',
        'compute_trainium/parcae_candidate.py','compute_trainium/parcae_stages_candidate.py',
        'compute_trainium/parcae_fp16_candidate.py','compute_trainium/parcae_optimizer_candidate.py',
        'compute_trainium/parcae_optimizer_precise.py','compute_trainium/parcae_optimizer_alignment.py',
        'compute_trainium/stable_recurrence_candidate.py','compute_trainium/parcae_training_candidate.py',
        'compute_trainium/parcae_work_accounting.py','compute_trainium/parcae_scientific_schedule.py',
        'compute_trainium/parcae_scientific_state.py','compute_trainium/depth_mixture.py',
        'compute_trainium/data.py','compute_trainium/metrics.py','compute_trainium/training.py',
        'compute_trainium/model.py','compute_trainium/titan.py','compute_trainium/queue.py',
        'compute_trainium/execution_isolation.py','compute_trainium/owned_native_phase.py',
        'compute_trainium/owned_native_compact.py','compute_trainium/native_comm_ports.py',
        'scripts/supervise_isolated.py','scripts/run_native_isolated.sh',
        'docs/LOW_DEPTH_TUNING_PROTOCOL_20261024.md',
        'tests/test_parcae_scientific_state.py','tests/test_parcae_scientific_schedule.py',
        'tests/test_low_depth_evaluation_identity.py','tests/test_native_comm_ports.py']
    sources={f:file_sha256(f) for f in files}
    cases={};identities=[]
    for lr_index,rate in enumerate(RATES):
        for arm in ARMS:
            row=next(r for r in preparation['records'] if r['arm']==arm);endpoint=row['tuning']
            key=f'{arm}-lr{lr_index}-s11'
            work=np.load(out/row['work_table'],mmap_mode='r')
            cases[key]={'arm':arm,'seed':11,'state_seed':2020,'initial_state':'s11-'+('ordinary' if arm=='ordinary' else 'family'),
                'parameters':row['parameters'],'peak_lr':rate,'warmup_targets':50_000_000,
                'cooldown_start_update':288,'cooldown_start_targets':endpoint['cooldown_start_targets'],
                'total_updates':576,'end_targets':endpoint['real_targets'],'depth_table':row['depth_table'],
                'depth_table_sha256':row['depth_table_sha256'],'work_table':row['work_table'],
                'work_table_sha256':row['work_table_sha256'],'checkpoint_updates':[20,576],
                'segment_ends':[20,576],'training_matmuls':int(work[:576].sum()),
                'primary_inference_depth':2 if arm=='poisson2' else 1}
            identities += [f'low-depth-tune-v2-{key}-{phase}' for phase in ['first','fit']]
    profile=json.loads(paths['model_training'].read_text())
    speeds={}
    for arm in ARMS:
        r=next(r for r in profile['records'] if r['arm']==arm and r['precision']=='fp16' and r['accumulation']==16)
        if not r['passes_recovery_gate'] or r['independent_trained_gradient']['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT':
            raise ValueError('Chosen precision/batch has incomplete training qualification')
        speeds[arm]=r['warm_targets_per_second_per_eight_chips']
    estimated=sum(c['end_targets']/speeds[c['arm']] for c in cases.values())/2
    evaluation=a.root/'variable-depth-fresh-v1/dev.json'
    plan={'identity':'parcae-low-depth-tuning-v2','registered_unix':time.time(),
        'purpose':'natural_low_depth_scientific_pilot','launch_gates_complete':True,
        'source_sha256':sources,'worker_source_sha256':{k:v for k,v in sources.items() if k.startswith(('compute_trainium/','scripts/','configs/'))},
        'qualification_receipts':{name:{'path':str(path),'sha256':file_sha256(path)} for name,path in paths.items()},
        'preparation_plan_sha256':file_sha256(out/'preparation-plan.json'),
        'model':preparation['model'],'initial_states':preparation['initial_states'],
        'optimizer_scalar_input_policy':'runtime_host_f32_tensor_copy','precision':'fp16','accumulation':16,'routing_seed':271828,'cases':cases,
        'root_comm_ports':reserve_ports(a.root,identities),'lr_grid':RATES,
        'data':{'corpus':'natural-append-6500m-v2/source.json','packed':'natural-append-6500m-v2/packed',
            'corpus_sha256':preparation['source_corpus_sha256'],'manifest_sha256':preparation['packed_manifest_sha256'],
            'packing_fingerprint':preparation['packing_fingerprint']},
        'evaluation':{'corpus':'variable-depth-fresh-v1/dev.json','corpus_sha256':file_sha256(evaluation),
                      'scope':'consumed_development_only'},
        'measured_eight_chip_targets_per_second':speeds,'estimated_retained_warm_native_seconds':estimated,
        'seconds_limit':10800,'phase_seconds_limit':1500,'cpu_score_seconds_limit':900,
        'latest_start_unix':dt.datetime(2026,10,25,2,0,tzinfo=dt.timezone.utc).timestamp(),
        'total_requested_training_matmuls':sum(c['training_matmuls'] for c in cases.values()),
        'scope':'Runtime-scalar v2; original v1 preserved before heldout scoring. Frozen four-rate opportunity per arm at301M natural targets, same fresh seed/backbone and source/depth/noise ordering. Own completed50% cooldown, absolute50M warmup and paired1:1 AdamW/Muon rate. CPU upstream FP32 TUNE scores select rates. No fresh confirmation, large main fit or terminal claim.'}
    write_state(plan_file,plan);write_state(public/'plan.json',plan)
    print(json.dumps({'identity':plan['identity'],'estimated_warm_hours':estimated/3600,'cases':len(cases)}))


if __name__=='__main__':main()
