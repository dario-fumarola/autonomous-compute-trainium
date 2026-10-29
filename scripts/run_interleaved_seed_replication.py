"""Conditional fresh-initialization development replication; no fresh panel."""
import argparse,copy,json,os,shutil,statistics,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.run_probability_screen import ScreenExecution

ID='interleaved-seed-replication-v1'
PUBLIC=Path('results/interleaved_workspace_20261029/seed-replication-v1')
SEEDS=(730201,730237)
OLD='interleaved-lr-tuning-v1'
MIXED='interleaved-mixed-round-tuning-v1'


def select_recipes(root):
    base=Path('results/interleaved_workspace_20261029');gates={}
    old=json.loads((base/'tuning-v1/analysis.json').read_text())['selection']
    mixed=json.loads((base/'mixed-round-tuning-v1/analysis.json').read_text())['selection']
    options=[dict(recipe=OLD,selection=old['ordinary'],tied=False)]
    for label,recipe in [('tied','interleaved-tied-ordinary-tuning-v1'),
                         ('frontier-deeper','interleaved-frontier-deeper-tuning-v1'),
                         ('frontier-wider','interleaved-frontier-wider-tuning-v1')]:
        f=base/(label+'-comparison-v1')/'report.json';r=json.loads(f.read_text())
        expected='COMPLETE_TIED_ORDINARY_PAIRED_COMPARISON' if label=='tied' else 'COMPLETE_ORDINARY_FRONTIER_PAIRED_COMPARISON'
        if r['status']!=expected or not r['useful_equal_data_screen'] or not r['selected_control']['bracketed']:
            raise AssertionError('Candidate must first survive every completed conventional control')
        gates[str(f)]=file_sha256(f);options.append(dict(recipe=recipe,selection=r['selected_control'],tied=label=='tied'))
    best=min(options,key=lambda x:(x['selection']['broad_nll'],x['recipe']))
    if not all(x['selection']['bracketed'] for x in options) or not mixed['bracketed'] or not old['pause']['bracketed']:
        raise AssertionError('Every chosen LR must be bracketed')
    return {'latent':dict(recipe=MIXED,selection=mixed,tied=False),
            'ordinary':best,'pause':dict(recipe=OLD,selection=old['pause'],tied=False)},gates


def one_trial(parent,role,seed,identity,selection):
    p=copy.deepcopy(parent);arm=copy.deepcopy(parent['arms'][selection['arm']])
    if seed in (parent['seed'],9901) or seed not in SEEDS:raise ValueError('New registered seed required')
    arm['phase']=identity+'-fit'
    p.update(identity=identity,seed=seed,arms={role:arm},evaluation_points=[],learning_rate_grid=[arm['lr']],
        automatic_promotion=False,scope='Fixed-recipe new-initialization development replication; no hyperparameter selection, fresh source confirmation or matched-work claim.')
    return p


def prepare(root):
    recipes,gates=select_recipes(root)
    if managed_running():raise RuntimeError('Prior Native owner must close')
    for f in Path('results').rglob('plan.json'):
        prior=json.loads(f.read_text())
        if prior.get('seed') in SEEDS:raise AssertionError('Selected initialization seed was already registered')
    if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3 or shutil.disk_usage(root).free<43*1024**3:
        raise RuntimeError('Protected reserves plus8GiB queue allowance')
    projection=project_snapshot_inodes(Path('.'),6)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('All six source snapshots must fit')
    source={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/INTERLEAVED_SEED_REPLICATION_20261029.md'):
        source[f]=file_sha256(f)
    worker,host=split_snapshot_sources(source);start=time.time();deadline=start+14400
    trials={};plans={}
    for seed in SEEDS:
        for role,recipe in recipes.items():
            parent_path=root/recipe['recipe']/'plan.json';parent=json.loads(parent_path.read_text())
            for f in ('scripts/train_interleaved_screen.py','compute_trainium/model.py','compute_trainium/interleaved_workspace.py',
                      'compute_trainium/grouped_interleaved_workspace.py','compute_trainium/interleaved_model_factory.py'):
                if file_sha256(f)!=parent['source_sha256'][f]:raise AssertionError('Qualified scientific implementation changed')
            key=f'{seed}-{role}';identity=ID+'-s'+key
            p=one_trial(parent,role,seed,identity,recipe['selection']);phase=p['arms'][role]['phase']
            p.update(registered_unix=start,absolute_deadline_unix=deadline,seconds_limit=14400,phase_seconds_limit=1800,cpu_phase_seconds_limit=1200,
                source_sha256=source,worker_source_sha256=worker,host_only_source_sha256=host,
                root_comm_ports=reserve_ports(root,[phase]),phases={phase:{}},inode_projection=projection,artifact_allowance_bytes=8*1024**3,
                parent_plan_sha256=file_sha256(parent_path))
            plans[key]=p;trials[key]=dict(identity=identity,role=role,seed=seed,tied=recipe['tied'],parent=recipe['recipe'],
                selected_arm=recipe['selection']['arm'],parent_plan_sha256=file_sha256(parent_path))
    reference=plans[f'{SEEDS[0]}-latent'];evaluation=root/reference['evaluation_relative'];er=json.loads((evaluation/'report.json').read_text())
    files={n:er['files'][n] for n in ('broad-tokens.npy','PRIVATE-broad-metadata.json')}
    if any(file_sha256(evaluation/n)!=h for n,h in files.items()):raise AssertionError('Frozen reused DEV changed')
    for key,p in plans.items():
        if any(p[k]!=reference[k] for k in ('data_relative','data_report_sha256','evaluation_relative','evaluation_report_sha256','world','batch','weight_decay')):
            raise AssertionError('Equal source exposure and objective recipe required')
        role=trials[key]['role'];a=p['arms'][role];b=reference['arms']['latent']
        if any(a[k]!=b[k] for k in ('real_targets','warmup_updates','decay_start','checkpoint_updates')) or len(a['depth_schedule'])!=len(b['depth_schedule']):
            raise AssertionError('Equal complete source schedule required')
        out=root/p['identity'];out.mkdir();write_state(out/'plan.json',p);trials[key]['plan_sha256']=file_sha256(out/'plan.json')
    parent=dict(identity=ID,registered_unix=start,absolute_deadline_unix=deadline,paired_seeds=list(SEEDS),recipes=recipes,trials=trials,
        source_sha256=source,gate_sha256=gates,inode_projection=projection,artifact_allowance_bytes=8*1024**3,
        evaluation_relative=reference['evaluation_relative'],evaluation_files=files,bootstrap_seed=10061,practical_gain_nats=.01,
        protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False,
        scope='Two new initialization seeds on reused DEV, fixed recipes. No fresh-panel confirmation, general reasoning, learned allocation, supported thought-budget scaling or training-efficiency claim.')
    (root/ID).mkdir();PUBLIC.mkdir(parents=True);write_state(root/ID/'plan.json',parent);write_state(PUBLIC/'plan.json',parent)


def run(root):
    from scripts.readout_interleaved_replication import run as readout
    start=time.time();out=root/ID;p=json.loads((out/'plan.json').read_text());result=dict(status='RUNNING_INTERLEAVED_SEED_REPLICATION',records={})
    try:
        for f,h in p['source_sha256'].items():
            if file_sha256(f)!=h:raise AssertionError('Frozen replication source changed')
        for f,h in p['gate_sha256'].items():
            if file_sha256(f)!=h:raise AssertionError('Frozen admission evidence changed')
        for key,t in p['trials'].items():
            child=root/t['identity']/'plan.json'
            if file_sha256(child)!=t['plan_sha256']:raise AssertionError('Frozen fit plan changed')
            e=ScreenExecution(root,child,root/t['identity']/'execution');role=t['role'];arm=e.plan['arms'][role];phase=arm['phase']
            records={}
            try:
                module='scripts.tied_interleaved_entry' if t['tied'] else 'scripts.train_interleaved_screen'
                extra=['--entry','train'] if t['tied'] else []
                records['fit']=e.native(phase,module,extra+['--root','/work/instance-nvme','--plan',f"/work/instance-nvme/{t['identity']}/plan.json",
                    '--plan-sha',e.sha,'--arm',role,'--comm-port',str(e.plan['root_comm_ports'][phase])],'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS')
                dest=root/t['identity']/('audit-'+role)
                module='scripts.tied_interleaved_entry' if t['tied'] else 'scripts.audit_interleaved_screen'
                extra=['--entry','scientific-audit'] if t['tied'] else []
                records['audit']=e.cpu(t['identity']+'-audit',module,extra+['--root',str(root),'--plan',str(child),'--plan-sha',e.sha,
                    '--arm',role,'--output',str(dest)],dest/'report.json','PASS_INTERLEAVED_SCREEN_MATURE_AUDITS')
                records['status']='COMPLETE_AUDITED_REPLICATION_FIT'
            finally:
                result['records'][key]=records;write_state(root/t['identity']/'report.json',records)
                e.close(root/t['identity']/'execution-costs.json');write_state(PUBLIC/'progress.json',result)
        for seed in SEEDS:
            if result['records'][f'{seed}-latent']['fit']['initial_model_digest']!=result['records'][f'{seed}-pause']['fit']['initial_model_digest']:
                raise AssertionError('Same-shape paired initialization mismatch')
        for seed in SEEDS:readout(root,out/'plan.json',seed)
        readouts={str(seed):json.loads((out/f'readout-{seed}/report.json').read_text()) for seed in SEEDS}
        gains={role:[readouts[str(s)]['comparisons'][role]['comparison']['gain'] for s in SEEDS] for role in ('ordinary','pause')}
        passed=all(readouts[str(s)]['comparisons'][role]['comparison']['gain']>=p['practical_gain_nats'] and
            readouts[str(s)]['comparisons'][role]['comparison']['conditional_95_interval'][0]>0 for s in SEEDS for role in gains)
        result.update(status='COMPLETE_INTERLEAVED_SEED_REPLICATION',readouts=readouts,paired_gains=gains,
            summary={role:dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v),seeds=len(v)) for role,v in gains.items()},
            reproducibility_screen_pass=passed,protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False,scope=p['scope'])
    except BaseException as error:
        result.update(status='STOPPED_INTERLEAVED_SEED_REPLICATION',error_type=type(error).__name__,error=str(error));raise
    finally:
        costs={key:json.loads((root/t['identity']/'execution-costs.json').read_text()) for key,t in p['trials'].items()
               if (root/t['identity']/'execution-costs.json').exists()}
        write_state(PUBLIC/'execution-costs.json',dict(trials=costs,scope='All admitted child fit/audit phases, including stopped work; exact CPU readout wall times are in the main report, with zero chips.'))
        result['wall_seconds']=time.time()-start;write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a.root)
