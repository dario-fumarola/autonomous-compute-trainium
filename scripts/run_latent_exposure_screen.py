"""One fixed exposure pilot with a conditional, fully specified work control."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID='latent-learning-exposure-v1'
SOURCE='latent-exposure-source-audit-v1'
PUBLIC=Path('results/latent_exposure_20261030/learning-v1')
SEED=730271
RECIPES={'latent':('interleaved-mixed-round-tuning-v1','mixed-lr1'),
         'ordinary':('interleaved-tied-ordinary-tuning-v1','tied-lr1'),
         'pause':('interleaved-lr-tuning-v1','pause-lr1')}


def read(p):return json.loads(p.read_text())


def trial_identity(key):
    if key not in ('ordinary','pause','latent','ordinary_work'):
        raise ValueError('Registered exposure role required')
    return ID+'-'+key.replace('_','-')


def arm_for(parent,name,role,updates,real_targets):
    from compute_trainium.model import ModelConfig
    from compute_trainium.interleaved_workspace import matrix_work
    from scripts.run_interleaved_mixed_tuning import mixed_schedule
    a=copy.deepcopy(parent['arms'][name]);c=ModelConfig(**parent['model_config'])
    if role not in ('latent','ordinary','pause') or updates<2:raise ValueError('Frozen supported role and endpoint required')
    schedule=mixed_schedule(updates) if role=='latent' else [a['depth_schedule'][0]]*updates
    a.update(depth_schedule=schedule,real_targets=real_targets,checkpoint_updates=[updates],
        recovery_checkpoint_updates=[math.floor(.9*updates)],decay_start=math.floor(.9*updates),
        nominal_matrix_flops=sum(64*matrix_work(c,512,k,mode=a['workspace_mode'],batch=2)['training_matmul_flops'] for k in schedule))
    if a['lr']!=.001 or a['warmup_updates']!=64:raise AssertionError('Preserve the independently selected recipe')
    return a


def advancement(broad,arithmetic):
    return all(broad['comparisons'][role]['comparison']['gain']>=.01
        and broad['comparisons'][role]['comparison']['conditional_95_interval'][0]>0
        and arithmetic['comparisons'][role]['answer']['mean']>=.03
        and arithmetic['comparisons'][role]['answer']['conditional_95_interval'][0]>0
        and arithmetic['comparisons'][role]['first']['mean']>=0 for role in ('ordinary','pause'))


def arithmetic_statistics(rows,seed=40531,replicates=2000):
    import numpy as np
    roles=('latent','ordinary','pause');bins=('2','3','4-6')
    cells={r:{v['index']:v for v in rows if v['role']==r} for r in roles};ref=cells['latent'];ids=set(ref)
    if not ids or len(rows)!=3*len(ids) or len({v['group'] for v in ref.values()})!=len(ids):raise AssertionError('Complete unique-source scoring required')
    for cell in cells.values():
        if set(cell)!=ids:raise AssertionError('All declared models and cases required')
        for i,v in cell.items():
            if any(v[k]!=ref[i][k] for k in ('group','stratum','targets','seed')):raise AssertionError('Source pairing changed')
            if any(not math.isfinite(v[k]) for k in ('answer_nll','first_nll','exact_prefix_correct')):raise AssertionError('Finite metrics required')
    indices={b:sorted(i for i in ids if ref[i]['stratum']==b) for b in bins}
    if any(len(v)<32 for v in indices.values()):raise AssertionError('Frozen minimum coverage required')
    rng=np.random.default_rng(seed);draws={b:rng.integers(len(v),size=(replicates,len(v))) for b,v in indices.items()}
    def values(role,field):return {b:np.array([cells[role][i][field] for i in indices[b]]) for b in bins}
    def estimate(v):
        bootstrap=sum(v[b][draws[b]].mean(1) for b in bins)/3
        return dict(mean=float(sum(v[b].mean() for b in bins)/3),bins={b:float(v[b].mean()) for b in bins},
            conditional_95_interval=np.quantile(bootstrap,[.025,.975]).tolist())
    means={r:{f:estimate(values(r,f)) for f in ('answer_nll','first_nll','exact_prefix_correct')} for r in roles}
    comparisons={}
    for role in ('ordinary','pause'):
        comparisons[role]={}
        for label,field in [('answer','answer_nll'),('first','first_nll')]:
            left=values(role,field);right=values('latent',field)
            comparisons[role][label]=estimate({b:left[b]-right[b] for b in bins})
    return dict(cases=len(ids),bins={b:len(v) for b,v in indices.items()},means=means,comparisons=comparisons,
        trained_seeds=1,scope='One fixed-recipe development seed; conditional source intervals are not training-seed confidence.')


def prepare(root):
    from compute_trainium.execution_isolation import managed_running
    from compute_trainium.native_comm_ports import reserve_ports
    from compute_trainium.native_inode_projection import project_snapshot_inodes
    from compute_trainium.native_snapshot_manifest import split_snapshot_sources
    from scripts.readout_interleaved_bracket_type import checkpoint
    if managed_running():raise RuntimeError('Close existing Native owner')
    out=root/ID
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every exposure attempt')
    sr=read(root/SOURCE/'report.json');sp=read(root/SOURCE/'plan.json')
    if sr['status']!='COMPLETE_EXPOSURE_SAFE_SOURCE_AUDIT' or any(sr[k] for k in ('source_overlap','expression_overlap','protected_overlap','model_scores')):
        raise AssertionError('Complete prospective source isolation required')
    if min(shutil.disk_usage(root).free-9*1024**3,shutil.disk_usage('/').free)<35*1024**3:
        raise RuntimeError('Protected reserves plus9GiB complete-queue artifacts')
    for f in Path('results').rglob('plan.json'):
        if read(f).get('seed')==SEED:raise AssertionError('New registered initialization seed required')
    projection=project_snapshot_inodes(Path('.'),4)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Four fit snapshots must preserve inode reserve')
    source={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/LATENT_LEARNING_EXPOSURE_20261030.md','tests/test_latent_exposure_screen.py'):
        source[f]=file_sha256(f)
    worker,host=split_snapshot_sources(source);now=time.time();deadline=now+43200
    trials={};evidence={};plans={}
    for key in ('ordinary','pause','latent','ordinary_work'):
        role='ordinary' if key=='ordinary_work' else key;recipe,name=RECIPES[role]
        parent_path=root/recipe/'plan.json';parent=read(parent_path)
        evidence[role]=checkpoint(root,recipe,name)
        for f in ('scripts/train_interleaved_screen.py','scripts/audit_interleaved_screen.py',
                'compute_trainium/interleaved_workspace.py','compute_trainium/grouped_interleaved_workspace.py',
                'compute_trainium/interleaved_model_factory.py','compute_trainium/interleaved_objective.py',
                'compute_trainium/indexed_natural_sources.py','compute_trainium/bounded_replica_capture.py'):
            if file_sha256(f)!=parent['source_sha256'][f]:raise AssertionError('Qualified numerical/training path changed')
        budget=sp['endpoints']['ordinary_matched_work' if key=='ordinary_work' else 'one_billion']
        identity=trial_identity(key);phase=identity+'-fit';arm=arm_for(parent,name,role,budget['updates'],budget['real_targets']);arm['phase']=phase
        if budget['updates']*128+128>=parent['audit_window_start']:raise AssertionError('Independent audit source must stay beyond all fit prefixes')
        p=copy.deepcopy(parent);p.update(identity=identity,seed=SEED,arms={role:arm},evaluation_points=[],learning_rate_grid=[arm['lr']],
            registered_unix=now,absolute_deadline_unix=deadline,seconds_limit=43200,
            phase_seconds_limit=21600 if key=='ordinary_work' else 7200,cpu_phase_seconds_limit=1800,
            source_sha256=source,worker_source_sha256=worker,host_only_source_sha256=host,
            root_comm_ports=reserve_ports(root,[phase]),phases={phase:{}},inode_projection=projection,
            artifact_allowance_bytes=9*1024**3,parent_plan_sha256=file_sha256(parent_path),automatic_promotion=False,
            scope='Fixed selected recipe at a new exposure; one new random seed, no LR selection. Equal-data mechanism test; longer ordinary is conditional. Not confirmation or a fully retuned frontier.')
        child=root/identity;child.mkdir();write_state(child/'plan.json',p);plans[key]=p
        trials[key]=dict(identity=identity,role=role,tied=role=='ordinary',seed=SEED,plan_sha256=file_sha256(child/'plan.json'))
    ref=plans['latent'];evaluation=root/ref['evaluation_relative'];er=read(evaluation/'report.json')
    files={n:er['files'][n] for n in ('broad-tokens.npy','PRIVATE-broad-metadata.json')}
    if any(file_sha256(evaluation/n)!=h for n,h in files.items()):raise AssertionError('Historical broad panel changed')
    if plans['ordinary_work']['arms']['ordinary']['nominal_matrix_flops']!=sp['maximum_training_matrix_work']:
        raise AssertionError('Source-audited matched-work exposure changed')
    if ref['arms']['latent']['nominal_matrix_flops']!=sp['candidate_training_matrix_work']:
        raise AssertionError('Candidate work changed after source audit')
    parent=dict(identity=ID,seed=SEED,registered_unix=now,absolute_deadline_unix=deadline,trials=trials,source_sha256=source,
        source_audit_sha256=file_sha256(root/SOURCE/'report.json'),source_plan_sha256=file_sha256(root/SOURCE/'plan.json'),
        panel_sha256=sr['files']['PRIVATE-panel.json'],qualified_evidence=evidence,inode_projection=projection,
        evaluation_relative=ref['evaluation_relative'],evaluation_files=files,bootstrap_seed=40531,
        arithmetic_bootstrap_replicates=2000,artifact_allowance_bytes=9*1024**3,scientific_fit_admitted=True,
        conditional_work_control=True,protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False,
        scope=Path('docs/LATENT_LEARNING_EXPOSURE_20261030.md').read_text())
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',parent);write_state(PUBLIC/'plan.json',parent)
    print(json.dumps(dict(status='REGISTERED_BOUNDED_LATENT_EXPOSURE',seed=SEED,updates=sp['endpoints'],conditional_work_control=True)),flush=True)


def arithmetic_readout(root,p,stage,selected):
    from scripts.readout_interleaved_bracket_type import checkpoint
    identity=ID+'-'+stage+'-arithmetic';out=root/identity;out.mkdir()
    points={}
    for role,key in selected.items():
        trial=p['trials'][key];point=checkpoint(root,trial['identity'],role);point.update(seed=SEED,role=role);points[role]=point
    start=time.time();rp=dict(identity=identity,points=points,registered_unix=start,
        absolute_deadline_unix=min(start+1800,p['absolute_deadline_unix']),threads=4,batch=8,
        maximum_worker_rss_bytes=6*1024**3,source_sha256=p['source_sha256'],
        panel_report_sha256=p['source_audit_sha256'],panel_files={'PRIVATE-panel.json':p['panel_sha256']})
    write_state(out/'plan.json',rp);write_state(PUBLIC/(stage+'-arithmetic-plan.json'),rp)
    jobs=[];logs=[]
    try:
        for key in points:
            log=(out/(key+'.log')).open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','scripts.run_latent_exposure_screen','--root',str(root),
                '--action','arithmetic-worker','--stage',stage,'--key',key],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
        from scripts.readout_direct_arithmetic import guard
        while any(j.poll() is None for j in jobs):
            guard(root,rp)
            if any(j.poll() not in (None,0) for j in jobs):raise RuntimeError('Exposure arithmetic worker failed')
            time.sleep(2)
        if any(j.returncode for j in jobs):raise RuntimeError('Exposure arithmetic worker failed')
        receipts={key:read(out/(key+'-report.json')) for key in points}
        if any(v['status']!='COMPLETE_DIRECT_ARITHMETIC_WORKER' or len(v['checks'])!=3 for v in receipts.values()):
            raise AssertionError('Complete arithmetic numerical checks required')
        rows=[v for key in points for v in read(out/('PRIVATE-'+key+'-scores.json'))]
        result=arithmetic_statistics(rows,p['bootstrap_seed'],p['arithmetic_bootstrap_replicates'])
        result.update(status='COMPLETE_EXPOSURE_ARITHMETIC_READOUT',workers=receipts,wall_seconds=time.time()-start,
            assigned_chips=0,optimizer_updates=0,protected_groups_scored=0)
        write_state(out/'report.json',result);write_state(PUBLIC/(stage+'-arithmetic.json'),result);return result
    finally:
        for j in jobs:
            if j.poll() is None:j.terminate()
        for j in jobs:
            try:j.wait(timeout=10)
            except subprocess.TimeoutExpired:j.kill();j.wait()
        for log in logs:log.close()


def readouts(root,p,stage,ordinary_key):
    from scripts.readout_interleaved_replication import run as broad_readout
    selected={'latent':'latent','ordinary':ordinary_key,'pause':'pause'}
    identity=ID+'-'+stage+'-broad';out=root/identity;out.mkdir()
    rp=dict(identity=identity,paired_seeds=[SEED],absolute_deadline_unix=p['absolute_deadline_unix'],
        source_sha256=p['source_sha256'],trials={f'{SEED}-{role}':p['trials'][key] for role,key in selected.items()},
        evaluation_relative=p['evaluation_relative'],evaluation_files=p['evaluation_files'],bootstrap_seed=p['bootstrap_seed'])
    write_state(out/'plan.json',rp);write_state(PUBLIC/(stage+'-broad-plan.json'),rp)
    broad_readout(root,out/'plan.json',SEED);b=read(out/f'readout-{SEED}/report.json')
    write_state(PUBLIC/(stage+'-broad.json'),b)
    return dict(broad=b,arithmetic=arithmetic_readout(root,p,stage,selected))


def run(root):
    from scripts.run_probability_screen import ScreenExecution
    out=root/ID;p=read(out/'plan.json');start=time.time();result=dict(status='RUNNING_LATENT_EXPOSURE',records={})
    def fit(key):
        trial=p['trials'][key];path=root/trial['identity']/'plan.json'
        if file_sha256(path)!=trial['plan_sha256']:raise AssertionError('Frozen child plan changed')
        e=ScreenExecution(root,path,root/trial['identity']/'execution');role=trial['role'];arm=e.plan['arms'][role];phase=arm['phase'];records={}
        try:
            module='scripts.tied_interleaved_entry' if trial['tied'] else 'scripts.train_interleaved_screen'
            extra=['--entry','train'] if trial['tied'] else []
            records['fit']=e.native(phase,module,extra+['--root','/work/instance-nvme','--plan',f"/work/instance-nvme/{trial['identity']}/plan.json",
                '--plan-sha',e.sha,'--arm',role,'--comm-port',str(e.plan['root_comm_ports'][phase])],'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS')
            module='scripts.tied_interleaved_entry' if trial['tied'] else 'scripts.audit_interleaved_screen'
            extra=['--entry','scientific-audit'] if trial['tied'] else [];dest=root/trial['identity']/('audit-'+role)
            records['audit']=e.cpu(trial['identity']+'-audit',module,extra+['--root',str(root),'--plan',str(path),'--plan-sha',e.sha,
                '--arm',role,'--output',str(dest)],dest/'report.json','PASS_INTERLEAVED_SCREEN_MATURE_AUDITS')
            records['status']='COMPLETE_AUDITED_EXPOSURE_FIT'
        finally:
            result['records'][key]=records;write_state(root/trial['identity']/'report.json',records)
            e.close(root/trial['identity']/'execution-costs.json');write_state(PUBLIC/'progress.json',result)
    try:
        if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Frozen exposure code changed')
        if file_sha256(root/SOURCE/'report.json')!=p['source_audit_sha256'] or file_sha256(root/SOURCE/'PRIVATE-panel.json')!=p['panel_sha256']:
            raise AssertionError('Prospective source isolation changed')
        for key in ('ordinary','pause','latent'):fit(key)
        if result['records']['latent']['fit']['initial_model_digest']!=result['records']['pause']['fit']['initial_model_digest']:
            raise AssertionError('Same-shape paired initialization changed')
        a=readouts(root,p,'equal-data','ordinary');result['equal_data']=a
        passed=advancement(a['broad'],a['arithmetic']);result['equal_data_development_gate']=passed
        write_state(PUBLIC/'progress.json',result)
        if passed:
            fit('ordinary_work');b=readouts(root,p,'matched-work','ordinary_work');result['matched_work']=b
            result['matched_work_development_gate']=advancement(b['broad'],b['arithmetic'])
        else:result['matched_work_control']='NOT_RUN_PREDECLARED_EQUAL_DATA_GATE_FAILED'
        result['status']='COMPLETE_BOUNDED_LATENT_EXPOSURE'
    except BaseException as error:
        result.update(status='STOPPED_BOUNDED_LATENT_EXPOSURE',error_type=type(error).__name__,error=str(error));raise
    finally:
        costs={k:read(root/t['identity']/'execution-costs.json') for k,t in p['trials'].items() if (root/t['identity']/'execution-costs.json').exists()}
        write_state(PUBLIC/'execution-costs.json',dict(trials=costs,scope='Every admitted Native and audit attempt; source preparation and exact CPU readout are separately retained.'))
        result.update(wall_seconds=time.time()-start,protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False,
            plan_sha256=file_sha256(out/'plan.json'))
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run','arithmetic-worker'],required=True);p.add_argument('--stage');p.add_argument('--key')
    a=p.parse_args()
    if a.action=='arithmetic-worker':
        from scripts import readout_direct_arithmetic as engine
        engine.ID=ID+'-'+a.stage+'-arithmetic';engine.PANEL=SOURCE;engine.worker(a.root,a.key)
    else:(prepare if a.action=='prepare' else run)(a.root)
