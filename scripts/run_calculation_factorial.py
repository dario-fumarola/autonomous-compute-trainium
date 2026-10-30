"""One source-exposure factorial; frozen numerical engines remain unchanged."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from compute_trainium.calculation_factorial import CELLS,arithmetic_statistics,broad_statistics,mechanism_gate
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID='calculation-exposure-factorial-v1'
PUBLIC=Path('results/calculation_exposure_20261030/factorial-v1')
SEED=730283
UPDATES=1957
DATA={'original':'indexed-natural-sources-v1','enriched':'calculation-exposure-index-v1-recovery-v1'}
PANEL='latent-exposure-source-audit-v1'
RECIPES={'ordinary':('interleaved-tied-ordinary-tuning-v1','tied-lr1'),
    'latent':('interleaved-mixed-round-tuning-v1','mixed-lr1')}


def read(path):return json.loads(Path(path).read_text())


def audit_start_for(condition,windows,updates=UPDATES):
    if condition not in DATA:raise ValueError('Registered source condition required')
    start=950000 if condition=='enriched' else 10000000
    if updates<1 or updates*128>=start or start+128>windows:
        raise AssertionError('Independent available audit source required')
    return start


def prepare(root):
    from compute_trainium.execution_isolation import managed_running
    from compute_trainium.indexed_natural_sources import IndexedNaturalSources
    from compute_trainium.native_comm_ports import reserve_ports
    from compute_trainium.native_inode_projection import project_snapshot_inodes
    from compute_trainium.native_snapshot_manifest import split_snapshot_sources
    from scripts.run_latent_exposure_screen import arm_for
    if managed_running():raise RuntimeError('Existing Native owner must close')
    closed=Path('results/latent_exposure_20261030/work-recovery-v1/report.json')
    if not closed.exists() or read(closed)['status']!='COMPLETE_LATENT_WORK_RECOVERY':
        raise RuntimeError('Close the original scientific comparison first')
    out=root/ID
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve all attempts')
    if min(shutil.disk_usage(root).free-6*1024**3,shutil.disk_usage('/').free)<35*1024**3:
        raise RuntimeError('Whole-queue artifact capacity plus protected reserve')
    projection=project_snapshot_inodes(Path('.'),4)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Four complete snapshot inode budget')
    for f in Path('results').rglob('plan.json'):
        if read(f).get('seed')==SEED:raise AssertionError('Fresh initialization required')
    source={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for name in ('scripts/run_native_isolated.sh','docs/CALCULATION_EXPOSURE_FACTORIAL_20261030.md',
                 'tests/test_calculation_factorial.py','tests/test_calculation_factorial_budget.py'):
        source[name]=file_sha256(name)
    worker,host=split_snapshot_sources(source)
    receipts={};exposures={};checked_raw={}
    for condition,name in DATA.items():
        report=read(root/name/'report.json')
        if report['status']!='PASS_INDEXED_NATURAL_SOURCES':raise AssertionError('Accepted original-token index required')
        for f,h in report['files'].items():
            if file_sha256(root/name/f)!=h:raise AssertionError('Source index changed')
        for domain,h in report['raw_token_sha256'].items():
            raw=root/report['source_relative']/(domain+'.u16')
            if raw not in checked_raw:checked_raw[raw]=file_sha256(raw)
            if checked_raw[raw]!=h:
                raise AssertionError('Original source token bytes changed')
        data=IndexedNaturalSources(root/name)
        exposures[condition]=data.target_count(0,UPDATES*128)
        audit_start=audit_start_for(condition,data.windows)
        receipts[condition]=dict(relative=name,report_sha256=file_sha256(root/name/'report.json'),
            files=report['files'],raw_token_sha256=report['raw_token_sha256'],audit_window_start=audit_start,
            real_targets=exposures[condition])
        del data
    if abs(exposures['original']/exposures['enriched']-1)>.001:
        raise AssertionError('Registered same-update target counts must agree within0.1%')
    panel=read(root/PANEL/'report.json')
    if panel['status']!='COMPLETE_EXPOSURE_SAFE_SOURCE_AUDIT' or any(panel[k] for k in ('source_overlap','expression_overlap','protected_overlap')):
        raise AssertionError('Audited harder panel required')
    pool=root/'calculation-document-pool-v1/report.json'
    pool_audit=Path('results/calculation_document_pool_20261030/source-v1/reaggregation.json')
    verified=read(pool_audit)
    # Binding the pool's original source-only exclusion receipt avoids a new
    # outcome-dependent expression filter in this experiment.
    if (verified['status']!='PASS_INDEPENDENT_POOL_REAGGREGATION'
            or verified['harder_panel_exact_expression_overlap']!=0 or verified['harder_panel_cases']!=259
            or verified['report_sha256']!=file_sha256(pool)):
        raise AssertionError('Enriched calculation pool must retain its hard-panel exclusions')
    now=time.time();deadline=now+14400;trials={};plans={}
    for key in CELLS:
        condition,role=key.split('-');recipe,name=RECIPES[role]
        base=root/recipe/'plan.json';parent=read(base)
        for f in ('scripts/train_interleaved_screen.py','scripts/audit_interleaved_screen.py',
                  'compute_trainium/interleaved_workspace.py','compute_trainium/grouped_interleaved_workspace.py',
                  'compute_trainium/indexed_natural_sources.py','compute_trainium/bounded_replica_capture.py'):
            if source[f]!=parent['source_sha256'][f]:raise AssertionError('Previously qualified numerical engine changed')
        identity=ID+'-'+key;phase=identity+'-fit'
        arm=arm_for(parent,name,role,UPDATES,exposures[condition]);arm['phase']=phase
        p=copy.deepcopy(parent);p.update(identity=identity,seed=SEED,arms={role:arm},evaluation_points=[],
            learning_rate_grid=[arm['lr']],registered_unix=now,absolute_deadline_unix=deadline,seconds_limit=14400,
            phase_seconds_limit=1800,cpu_phase_seconds_limit=1200,data_relative=DATA[condition],
            data_report_sha256=receipts[condition]['report_sha256'],audit_window_start=receipts[condition]['audit_window_start'],
            source_sha256=source,worker_source_sha256=worker,host_only_source_sha256=host,
            root_comm_ports=reserve_ports(root,[phase]),phases={phase:{}},inode_projection=projection,
            artifact_allowance_bytes=6*1024**3,parent_plan_sha256=file_sha256(base),automatic_promotion=False,
            scope='Fresh paired seed; original versus authentic calculation-enriched sources; unchanged uniform NTP and numerical engine. Equal updates, not matched training work.')
        child=root/identity;child.mkdir();write_state(child/'plan.json',p);plans[key]=p
        trials[key]=dict(identity=identity,role=role,tied=role=='ordinary',seed=SEED,condition=condition,
            plan_sha256=file_sha256(child/'plan.json'))
    ref=plans[CELLS[0]];evaluation=root/ref['evaluation_relative'];er=read(evaluation/'report.json')
    ef={n:er['files'][n] for n in ('broad-tokens.npy','PRIVATE-broad-metadata.json')}
    if any(file_sha256(evaluation/n)!=h for n,h in ef.items()):raise AssertionError('Original broad panel changed')
    p=dict(identity=ID,seed=SEED,registered_unix=now,absolute_deadline_unix=deadline,trials=trials,
        source_sha256=source,data_receipts=receipts,evaluation_relative=ref['evaluation_relative'],evaluation_files=ef,
        panel_relative=PANEL,panel_report_sha256=file_sha256(root/PANEL/'report.json'),
        panel_files={'PRIVATE-panel.json':panel['files']['PRIVATE-panel.json']},
        calculation_pool_report_sha256=file_sha256(root/'calculation-document-pool-v1/report.json'),
        calculation_pool_reaggregation_sha256=file_sha256(pool_audit),
        closed_work_report_sha256=file_sha256(closed),bootstrap_seed=40532,bootstrap_replicates=2000,
        threads=8,batch=8,maximum_worker_rss_bytes=6*1024**3,artifact_allowance_bytes=6*1024**3,
        protected_groups_scored=0,terminal_alpha_spent=0,automatic_promotion=False,scientific_fit_admitted=True,
        scope=Path('docs/CALCULATION_EXPOSURE_FACTORIAL_20261030.md').read_text())
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)
    print(json.dumps(dict(status='REGISTERED_CALCULATION_EXPOSURE_FACTORIAL',seed=SEED,updates_per_fit=UPDATES,targets=exposures)),flush=True)


def broad_worker(root,key):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.tied_interleaved_ordinary import make_model as tied_model
    from compute_trainium.interleaved_sequential_reference import cached_sequential
    from scripts.readout_direct_arithmetic import guard
    out=root/(ID+'-readout');p=read(out/'plan.json');point=p['points'][key];start=time.time();guard(root,p)
    if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Frozen readout code changed')
    if file_sha256(root/point['plan_relative'])!=point['plan_sha256'] or file_sha256(root/point['checkpoint_relative'])!=point['checkpoint_sha256']:
        raise AssertionError('Audited model changed')
    evaluation=root/p['evaluation_relative']
    if any(file_sha256(evaluation/n)!=h for n,h in p['evaluation_files'].items()):raise AssertionError('Frozen DEV changed')
    torch.set_num_threads(p['threads']);torch.set_num_interop_threads(1)
    fit=read(root/point['plan_relative']);role=point['role'];model=(tied_model(fit) if role=='ordinary' else make_model(fit)).double().eval()
    saved=torch.load(root/point['checkpoint_relative'],map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(saved['model']);del saved
    tokens=np.load(evaluation/'broad-tokens.npy',mmap_mode='r');meta=read(evaluation/'PRIVATE-broad-metadata.json')
    if len(tokens)!=384 or len(meta)!=384:raise AssertionError('Unchanged complete broad panel required')
    rows=[];checks=[]
    with torch.inference_mode():
        for i in (0,128,256):
            x=torch.from_numpy(np.array(tokens[i:i+1,:32],dtype=np.int64))
            a=cached_sequential(model,x) if role=='latent' else model(x,mode='ordinary')
            b=model.sequential(x) if role=='latent' else model(x,mode='ordinary')
            error=float((a.log_softmax(-1)-b.log_softmax(-1)).abs().max())
            if error>1e-8:raise AssertionError('Exact cached/full-prefix discrepancy')
            checks.append(dict(index=i,maximum_log_probability_error=error))
        for left in range(0,len(meta),2):
            guard(root,p);indices=list(range(left,left+2))
            x=torch.from_numpy(np.array(tokens[indices,:-1],dtype=np.int64));y=torch.from_numpy(np.array(tokens[indices,1:],dtype=np.int64))
            logits=cached_sequential(model,x) if role=='latent' else model(x,mode='ordinary')
            losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
            for j,i in enumerate(indices):
                m=meta[i];rows.append(dict(label=key,index=i,id=m['id'],group=m['group'],domain=m['domain'],targets=512,loss_sum=float(losses[j].sum())))
            del logits,losses
            if left%16==0:write_state(out/(key+'-broad-progress.json'),dict(completed=left+2,wall_seconds=time.time()-start))
    write_state(out/('PRIVATE-'+key+'-broad.json'),rows)
    write_state(out/(key+'-broad-report.json'),dict(status='COMPLETE_FACTORIAL_BROAD_WORKER',rows=len(rows),checks=checks,
        wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        assigned_chips=0,optimizer_updates=0))


def readouts(root,p):
    from scripts.readout_interleaved_bracket_type import checkpoint
    from scripts.readout_direct_arithmetic import guard
    out=root/(ID+'-readout');out.mkdir();start=time.time();rp=copy.deepcopy(p);rp['points']={}
    rp['absolute_deadline_unix']=min(start+3600,p['absolute_deadline_unix'])
    for key,t in p['trials'].items():
        point=checkpoint(root,t['identity'],t['role']);point.update(seed=SEED,role=t['role']);rp['points'][key]=point
    write_state(out/'plan.json',rp);write_state(PUBLIC/'readout-plan.json',rp)
    jobs=[];logs=[]
    try:
        for kind in ('broad','arithmetic'):
            jobs=[]
            for key in CELLS:
                log=(out/(key+'-'+kind+'.log')).open('w');logs.append(log)
                jobs.append(subprocess.Popen([sys.executable,'-m','scripts.run_calculation_factorial','--root',str(root),
                    '--action','worker','--key',key,'--kind',kind],stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1'}))
            while any(j.poll() is None for j in jobs):
                guard(root,rp)
                if any(j.poll() not in (None,0) for j in jobs):raise RuntimeError('Bounded exact readout failed')
                time.sleep(2)
            if any(j.returncode for j in jobs):raise RuntimeError('Bounded exact readout failed')
        br={k:read(out/('PRIVATE-'+k+'-broad.json')) for k in CELLS}
        broad=broad_statistics(br,p['bootstrap_replicates'],p['bootstrap_seed'])
        rows=[dict(v,cell=k) for k in CELLS for v in read(out/('PRIVATE-'+k+'-scores.json'))]
        arithmetic=arithmetic_statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed'])
        result=dict(status='COMPLETE_CALCULATION_FACTORIAL_READOUT',broad=broad,arithmetic=arithmetic,
            mechanism_screen_gate=mechanism_gate(arithmetic,broad['enriched_architecture']['comparison']['gain']),
            workers={k:{'broad':read(out/(k+'-broad-report.json')),'arithmetic':read(out/(k+'-report.json'))} for k in CELLS},
            wall_seconds=time.time()-start,assigned_chips=0,optimizer_updates=0,protected_groups_scored=0)
        if any(v['broad']['status']!='COMPLETE_FACTORIAL_BROAD_WORKER' or v['arithmetic']['status']!='COMPLETE_DIRECT_ARITHMETIC_WORKER'
               or len(v['broad']['checks'])!=3 or len(v['arithmetic']['checks'])!=3 for v in result['workers'].values()):
            raise AssertionError('Complete numerical receipts required')
        write_state(out/'report.json',result);write_state(PUBLIC/'readout.json',result);return result
    finally:
        for j in jobs:
            if j.poll() is None:j.terminate()
        for j in jobs:
            try:j.wait(timeout=10)
            except subprocess.TimeoutExpired:j.kill();j.wait()
        for log in logs:log.close()


def run(root):
    from scripts.run_probability_screen import ScreenExecution
    out=root/ID;p=read(out/'plan.json');start=time.time();result=dict(status='STOPPED_CALCULATION_EXPOSURE_FACTORIAL',records={})
    try:
        if any(file_sha256(f)!=h for f,h in p['source_sha256'].items()):raise AssertionError('Frozen experiment source changed')
        for condition,d in p['data_receipts'].items():
            if file_sha256(root/d['relative']/'report.json')!=d['report_sha256']:raise AssertionError('Accepted index changed')
        for key in CELLS:
            t=p['trials'][key];path=root/t['identity']/'plan.json'
            if file_sha256(path)!=t['plan_sha256']:raise AssertionError('Frozen fit changed')
            e=ScreenExecution(root,path,root/t['identity']/'execution');role=t['role'];phase=e.plan['arms'][role]['phase'];rec={}
            try:
                module='scripts.tied_interleaved_entry' if t['tied'] else 'scripts.train_interleaved_screen'
                extra=['--entry','train'] if t['tied'] else []
                rec['fit']=e.native(phase,module,extra+['--root','/work/instance-nvme','--plan',f"/work/instance-nvme/{t['identity']}/plan.json",
                    '--plan-sha',e.sha,'--arm',role,'--comm-port',str(e.plan['root_comm_ports'][phase])],
                    'COMPLETE_INTERLEAVED_SCREEN_FIT_PENDING_AUDITS')
                module='scripts.tied_interleaved_entry' if t['tied'] else 'scripts.audit_interleaved_screen'
                extra=['--entry','scientific-audit'] if t['tied'] else [];dest=root/t['identity']/('audit-'+role)
                rec['audit']=e.cpu(t['identity']+'-audit',module,extra+['--root',str(root),'--plan',str(path),
                    '--plan-sha',e.sha,'--arm',role,'--output',str(dest)],dest/'report.json','PASS_INTERLEAVED_SCREEN_MATURE_AUDITS')
                rec['status']='COMPLETE_AUDITED_FACTORIAL_FIT'
            finally:
                result['records'][key]=rec;write_state(root/t['identity']/'report.json',rec)
                e.close(root/t['identity']/'execution-costs.json');write_state(PUBLIC/'progress.json',result)
        for role in ('ordinary','latent'):
            if result['records']['original-'+role]['fit']['initial_model_digest']!=result['records']['enriched-'+role]['fit']['initial_model_digest']:
                raise AssertionError('Source intervention requires identical within-architecture initialization')
        result['readout']=readouts(root,p);result['status']='COMPLETE_CALCULATION_EXPOSURE_FACTORIAL'
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        costs={k:read(root/t['identity']/'execution-costs.json') for k,t in p['trials'].items() if (root/t['identity']/'execution-costs.json').exists()}
        write_state(PUBLIC/'execution-costs.json',dict(trials=costs))
        result.update(wall_seconds=time.time()-start,protected_groups_scored=0,terminal_alpha_spent=0,
            automatic_promotion=False,plan_sha256=file_sha256(out/'plan.json'))
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--action',choices=('prepare','run','worker'),required=True)
    parser.add_argument('--key',choices=CELLS);parser.add_argument('--kind',choices=('broad','arithmetic'))
    a=parser.parse_args()
    if a.action=='worker':
        if a.kind=='broad':broad_worker(a.root,a.key)
        else:
            from scripts import readout_direct_arithmetic as engine
            engine.ID=ID+'-readout';engine.PANEL=PANEL;engine.worker(a.root,a.key)
    else:(prepare if a.action=='prepare' else run)(a.root)
