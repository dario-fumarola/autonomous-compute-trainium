"""Frozen paired analysis for either independently tuned ordinary shape."""
import argparse,json,time
from pathlib import Path
from compute_trainium.model import ModelConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.compare_interleaved_tied import compare,inference

MIXED='interleaved-mixed-round-tuning-v1'
SEQUENTIAL='interleaved-mixed-sequential-dev-v1'


def locations(root,shape):
    if shape not in ('deeper','wider'):raise ValueError('Pre-inventoried shape required')
    tuning=f'interleaved-frontier-{shape}-tuning-v1'
    identity=f'interleaved-frontier-{shape}-comparison-v1'
    public=Path('results/interleaved_workspace_20261029')/f'frontier-{shape}-comparison-v1'
    return tuning,identity,public


def validate_exposure(candidate_plan,control_plan,candidate_arm,control_arm):
    for key in ('seed','world','batch','weight_decay','data_relative','data_report_sha256',
                'evaluation_relative','evaluation_report_sha256','audit_window_start'):
        if candidate_plan[key]!=control_plan[key]:raise AssertionError('Unequal comparison field: '+key)
    for key in ('real_targets','warmup_updates','decay_start','checkpoint_updates'):
        if candidate_arm[key]!=control_arm[key]:raise AssertionError('Unequal schedule field: '+key)
    if len(candidate_arm['depth_schedule'])!=len(control_arm['depth_schedule']):
        raise AssertionError('Unequal source-update count')


def prepare(root,shape):
    tuning,identity,public=locations(root,shape)
    if (root/tuning/'analysis/report.json').exists():raise AssertionError('Register before challenger quality')
    tp=root/tuning/'plan.json';sp=root/SEQUENTIAL/'plan.json';sr=root/SEQUENTIAL/'report.json'
    if json.loads(sr.read_text())['status']!='COMPLETE_SEQUENTIAL_DEV_READOUT':raise AssertionError('Complete candidate required')
    artifacts=[tp,sp,sr,root/MIXED/'plan.json',*sorted((root/SEQUENTIAL).glob('PRIVATE-worker*-scores.json'))]
    source_files=('scripts/compare_interleaved_frontier.py','scripts/compare_interleaved_tied.py',
                  'scripts/analyze_probability_screen.py','compute_trainium/model.py',
                  'docs/INTERLEAVED_FRONTIER_COMPARISON_20261029.md')
    plan=dict(identity=identity,shape=shape,registered_unix=time.time(),
        frozen_artifact_sha256={str(f.relative_to(root)):file_sha256(f) for f in artifacts},
        source_sha256={f:file_sha256(f) for f in source_files},
        bootstrap_seed=10023,bootstrap_replicates=2000,practical_gain_nats=.01,seconds_limit=900,
        absolute_deadline_unix=json.loads(tp.read_text())['absolute_deadline_unix']+900,
        scope='One selected training seed on reused DEV, equal authentic exposure. No fresh confirmation, general reasoning, allocation, exhaustive frontier or matched-training-work claim.')
    (root/identity).mkdir();public.mkdir(parents=True)
    write_state(root/identity/'plan.json',plan);write_state(public/'plan.json',plan)


def run(root,shape):
    start=time.time();tuning,identity,public=locations(root,shape)
    p=json.loads((root/identity/'plan.json').read_text())
    if start>p['absolute_deadline_unix']:raise RuntimeError('Immutable analysis deadline')
    for f,h in p['source_sha256'].items():
        if file_sha256(f)!=h:raise AssertionError('Frozen source changed')
    for f,h in p['frozen_artifact_sha256'].items():
        if file_sha256(root/f)!=h:raise AssertionError('Frozen candidate or training plan changed')
    control_public=Path('results/interleaved_workspace_20261029')/f'frontier-{shape}-tuning-v1'
    report=json.loads((control_public/'report.json').read_text())
    analysis=json.loads((control_public/'analysis.json').read_text())
    if report['status']!='COMPLETE_ORDINARY_FRONTIER_TUNING' or analysis['status']!='COMPLETE_ORDINARY_FRONTIER_TUNING_ANALYSIS':
        raise AssertionError('All control fits, audits and selection required')
    selected=analysis['selection'];folder=root/'runs'/(tuning+'-evaluation')
    if json.loads((folder/'report.json').read_text())['status']!='COMPLETE_INTERLEAVED_SCREEN_EVALUATION':
        raise AssertionError('Independent readout checks required')
    files=[folder/f'PRIVATE-rank{r}-scores.json' for r in range(64)]
    control=[row for f in files for row in json.loads(f.read_text()) if row['label']==selected['arm']+'-main' and row['panel']=='broad']
    candidate=[row for f in sorted((root/SEQUENTIAL).glob('PRIVATE-worker*-scores.json')) for row in json.loads(f.read_text())]
    result=compare(candidate,control,seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates'])
    mp=json.loads((root/MIXED/'plan.json').read_text());tp=json.loads((root/tuning/'plan.json').read_text())
    sp=json.loads((root/SEQUENTIAL/'plan.json').read_text());ca=mp['arms'][sp['selected_arm']];ta=tp['arms'][selected['arm']]
    validate_exposure(mp,tp,ca,ta)
    resources={}
    for name,plan,arm,latent in [('candidate',mp,ca,True),('control',tp,ta,False)]:
        receipt=json.loads((root/'runs'/arm['phase']/'report.json').read_text())
        resources[name]=dict(parameters=plan['parameters'],real_targets=arm['real_targets'],
            nominal_training_matrix_flops=arm['nominal_matrix_flops'],native_fit_wall_seconds=receipt['wall_seconds'],
            native_fit_assigned_chip_hours=16*receipt['wall_seconds']/3600,
            runtime_scope='Descriptive retained Native fit including startup, compile, audit and checkpoint; cross-launch, not hardware-time matched.',
            inference=inference(ModelConfig(**plan['model_config']),latent))
    result.update(status='COMPLETE_ORDINARY_FRONTIER_PAIRED_COMPARISON',shape=shape,selected_control=selected,
        candidate_arm=sp['selected_arm'],resources=resources,
        candidate_over_control_training_work=ca['nominal_matrix_flops']/ta['nominal_matrix_flops'],
        useful_equal_data_screen=bool(selected['bracketed'] and result['comparison']['gain']>=p['practical_gain_nats'] and result['comparison']['conditional_95_interval'][0]>0),
        control_score_sha256={str(f.relative_to(root)):file_sha256(f) for f in files},
        control_analysis_sha256=file_sha256(control_public/'analysis.json'),trained_seeds=1,protected_groups_scored=0,
        assigned_chips=0,optimizer_updates=0,automatic_promotion=False,wall_seconds=time.time()-start,scope=p['scope'])
    if time.time()-start>p['seconds_limit'] or time.time()>p['absolute_deadline_unix']:raise RuntimeError('Bounded analysis')
    write_state(root/identity/'report.json',result);write_state(public/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shape',choices=['deeper','wider'],required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a.root,a.shape)
