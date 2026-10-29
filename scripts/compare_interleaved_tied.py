"""Prospective paired comparison against the independently tuned challenger."""
import argparse,json,math,time
from pathlib import Path
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from compute_trainium.model import ModelConfig
from scripts.analyze_probability_screen import paired,score

ID='interleaved-tied-comparison-v1'
TIED='interleaved-tied-ordinary-tuning-v1'
MIXED='interleaved-mixed-round-tuning-v1'
SEQUENTIAL='interleaved-mixed-sequential-dev-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/tied-comparison-v1')


def validate_rows(rows):
    if len(rows)!=384 or len({x['id'] for x in rows})!=384 or {x['index'] for x in rows}!=set(range(384)):
        raise AssertionError('Exactly384 unique frozen views required')
    if any(sum(x['domain']==d for x in rows)!=128 for d in ('web','math','code')):
        raise AssertionError('Exact domain counts required')
    if any(x['targets']!=512 or not math.isfinite(x['loss_sum']) or x['loss_sum']<0 for x in rows):
        raise AssertionError('Finite loss on every original target required')


def compare(candidate,control,seed=10023,replicates=2000):
    validate_rows(candidate);validate_rows(control)
    left={x['id']:x for x in control};right={x['id']:x for x in candidate}
    if left.keys()!=right.keys():raise AssertionError('Frozen examples differ')
    if any(tuple(left[k][f] for f in ('index','domain','group','targets'))!=tuple(right[k][f] for f in ('index','domain','group','targets')) for k in left):
        raise AssertionError('Pairing metadata differs')
    broad=paired(control,candidate,seed=seed,replicates=replicates,weighted=True)
    domains={d:paired([r for r in control if r['domain']==d],[r for r in candidate if r['domain']==d],seed=seed,replicates=replicates) for d in ('web','math','code')}
    scores={name:{d:score([r for r in rows if r['domain']==d]) for d in domains} for name,rows in [('candidate',candidate),('control',control)]}
    return dict(comparison=broad,domains=domains,scores={n:dict(domains=s,broad_nll=sum(w*s[d] for d,w in zip(('web','math','code'),(.6,.2,.2)))) for n,s in scores.items()})


def inference(c,latent):
    n=512;slots=n*(2 if latent else 1)
    body=c.flops(0,slots)['forward_matmul_flops']
    full_attention=4*c.layers*c.dim*slots**2
    causal_attention=4*c.layers*c.dim*slots*(slots+1)//2
    return dict(source_positions=n,stored_positions=slots,
        nominal_cached_forward_matmul_flops=body-full_attention+causal_attention+2*n*c.dim*c.vocab_size,
        fp32_kv_bytes=2*c.layers*slots*c.kv_heads*(c.dim//c.heads)*4,
        note='Analytic cached causal matrix work only, FMA2; excludes elementwise, transfers, dispatch and measured latency. No serving-performance claim.')


def prepare(root):
    tied=root/TIED/'plan.json'
    if (root/TIED/'analysis/report.json').exists():raise AssertionError('Register before challenger quality is available')
    sp=root/SEQUENTIAL/'plan.json';sr=root/SEQUENTIAL/'report.json';s=json.loads(sr.read_text())
    if s['status']!='COMPLETE_SEQUENTIAL_DEV_READOUT':raise AssertionError('Complete exact sequential candidate required')
    files={str(f.relative_to(root)):file_sha256(f) for f in [tied,sp,sr,root/MIXED/'plan.json',*sorted((root/SEQUENTIAL).glob('PRIVATE-worker*-scores.json'))]}
    sources={f:file_sha256(f) for f in ('scripts/compare_interleaved_tied.py','scripts/analyze_probability_screen.py','compute_trainium/model.py','docs/INTERLEAVED_TIED_COMPARISON_20261029.md')}
    p=dict(identity=ID,registered_unix=time.time(),frozen_artifact_sha256=files,source_sha256=sources,
        bootstrap_seed=10023,bootstrap_replicates=2000,practical_gain_nats=.01,seconds_limit=900,absolute_deadline_unix=json.loads(tied.read_text())['absolute_deadline_unix']+900,
        candidate_selection='Previously frozen mixed LR selected by K2 NLL; exact sequential readout only.',
        control_selection='Independently selected lowest broad NLL among all three registered tied LRs; require bracketed optimum before advancing.',
        scope='One-seed reused-DEV equal-data comparison. No fresh confirmation, general reasoning, allocation, full conventional frontier or matched-training-work claim.')
    (root/ID).mkdir();PUBLIC.mkdir(parents=True);write_state(root/ID/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def run(root):
    start=time.time();p=json.loads((root/ID/'plan.json').read_text())
    if start>p['absolute_deadline_unix']:raise RuntimeError('Immutable analysis deadline')
    for f,h in p['source_sha256'].items():
        if file_sha256(f)!=h:raise AssertionError('Frozen analysis source changed')
    for f,h in p['frozen_artifact_sha256'].items():
        if file_sha256(root/f)!=h:raise AssertionError('Frozen candidate, panel rows or training plan changed')
    public=Path('results/interleaved_workspace_20261029/tied-ordinary-tuning-v1')
    report=json.loads((public/'report.json').read_text());analysis=json.loads((public/'analysis.json').read_text())
    if report['status']!='COMPLETE_TIED_ORDINARY_TUNING' or analysis['status']!='COMPLETE_TIED_ORDINARY_TUNING_ANALYSIS':
        raise AssertionError('All challenger fits,audits and common selection required')
    selected=analysis['selection'];folder=root/'runs'/(TIED+'-evaluation')
    if json.loads((folder/'report.json').read_text())['status']!='COMPLETE_INTERLEAVED_SCREEN_EVALUATION':raise AssertionError('Independent readout checks required')
    control_files=[folder/f'PRIVATE-rank{rank}-scores.json' for rank in range(64)]
    control=[r for f in control_files for r in json.loads(f.read_text()) if r['label']==selected['arm']+'-main' and r['panel']=='broad']
    candidate=[r for f in sorted((root/SEQUENTIAL).glob('PRIVATE-worker*-scores.json')) for r in json.loads(f.read_text())]
    result=compare(candidate,control,seed=p['bootstrap_seed'],replicates=p['bootstrap_replicates'])
    mp=json.loads((root/MIXED/'plan.json').read_text());tp=json.loads((root/TIED/'plan.json').read_text());sp=json.loads((root/SEQUENTIAL/'plan.json').read_text())
    candidate_arm=sp['selected_arm'];ca=mp['arms'][candidate_arm];ta=tp['arms'][selected['arm']]
    if ca['real_targets']!=ta['real_targets']:raise AssertionError('Equal authentic source exposure required')
    resources={}
    for label,plan,arm,latent in [('candidate',mp,ca,True),('control',tp,ta,False)]:
        receipt=json.loads((root/'runs'/arm['phase']/'report.json').read_text())
        resources[label]=dict(parameters=plan['parameters'],real_targets=arm['real_targets'],nominal_training_matrix_flops=arm['nominal_matrix_flops'],
            native_fit_wall_seconds=receipt['wall_seconds'],native_fit_assigned_chip_hours=16*receipt['wall_seconds']/3600,
            runtime_scope='Descriptive retained Native fit path, including its startup/compile/audit/checkpoint work. Cross-launch, not a hardware-time-matched experiment.',
            inference=inference(ModelConfig(**plan['model_config']),latent))
    result.update(status='COMPLETE_TIED_ORDINARY_PAIRED_COMPARISON',selected_control=selected,candidate_arm=candidate_arm,resources=resources,
        candidate_over_control_training_work=ca['nominal_matrix_flops']/ta['nominal_matrix_flops'],
        useful_equal_data_screen=(selected['bracketed'] and result['comparison']['gain']>=p['practical_gain_nats'] and result['comparison']['conditional_95_interval'][0]>0),
        control_score_sha256={str(f.relative_to(root)):file_sha256(f) for f in control_files},
        control_analysis_sha256=file_sha256(public/'analysis.json'),trained_seeds=1,protected_groups_scored=0,assigned_chips=0,optimizer_updates=0,
        automatic_promotion=False,wall_seconds=time.time()-start,scope=p['scope'])
    if time.time()-start>p['seconds_limit'] or time.time()>p['absolute_deadline_unix']:raise RuntimeError('Analysis time bound')
    write_state(root/ID/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a.root)
