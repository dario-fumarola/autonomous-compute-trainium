"""One qualified budget-cue fit; all old controls remain frozen and visible."""
import argparse,copy,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_probability_screen import ScreenExecution

ID='probability-budget-cue-screen-v1'
PARENT='probability-fixed-one-v1'
PUBLIC=Path('results/probability_workspace_20261029/budget-cue-screen-v1')


def prepare(root):
    if managed_running():raise RuntimeError('Complete qualification first')
    gate=Path('results/probability_workspace_20261029/budget-cue-warm-v1/report.json')
    if json.loads(gate.read_text())['status']!='PASS_BUDGET_CUE_ACTUAL_SHAPE':raise AssertionError('Full actual-shape qualification required')
    prior=Path('results/probability_workspace_20261029/fixed-one-v1/report.json')
    if json.loads(prior.read_text())['status']!='COMPLETE_FIXED_ONE_SCREEN':raise AssertionError('Completed previous comparison required')
    p=copy.deepcopy(json.loads((root/PARENT/'plan.json').read_text()))
    qualified=json.loads((root/'probability-budget-cue-warm-v1/plan.json').read_text())
    for name in ('compute_trainium/budget_conditioned_workspace.py','compute_trainium/probability_model_factory.py',
                 'compute_trainium/probability_workspace.py','compute_trainium/probability_workspace_data.py',
                 'scripts/train_probability_screen.py','scripts/audit_probability_screen.py','scripts/evaluate_probability_screen.py'):
        if file_sha256(name)!=qualified['source_sha256'][name]:raise AssertionError('Qualified implementation changed')
    for relative,key in ((p['data_relative'],'data_report_sha256'),(p['evaluation_relative'],'evaluation_report_sha256')):
        base=root/relative
        if file_sha256(base/'report.json')!=p[key]:raise AssertionError('Frozen source receipt changed')
        receipt=json.loads((base/'report.json').read_text())
        if any(file_sha256(base/n)!=h for n,h in receipt['files'].items()):raise AssertionError('Frozen data bytes changed')
    projection=project_snapshot_inodes(Path('.'),2)
    if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Two snapshots plus protected inode reserve')
    if shutil.disk_usage('/').free<35*1024**3 or shutil.disk_usage(root).free<38*1024**3:raise RuntimeError('3GiB artifacts plus35GiB reserve')
    arm=copy.deepcopy(p['arms']['sampled']);final=arm['checkpoint_updates'][-1]
    arm.update(phase=ID+'-workspace',checkpoint_updates=[final])
    p['arms']={'workspace':arm}
    p['evaluation_points']=[dict(label='cue-final',arm='workspace',update=final,
        variants=[dict(label='k'+str(k),depth=k,proposal='expected') for k in (0,1,2)]+
                 [dict(label='permuted-k2',depth=2,proposal='expected',permute=True)])]
    oldeval=root/'runs'/(PARENT+'-evaluation')
    if json.loads((oldeval/'report.json').read_text())['status']!='COMPLETE_PROBABILITY_SCREEN_EVALUATION':raise AssertionError('Audited control readouts required')
    oldfit=json.loads((root/PARENT/'plan.json').read_text())
    initial_digests=set();reused={}
    for name,cfg in oldfit['arms'].items():
        receipt=root/PARENT/('audit-'+name)/'report.json';r=json.loads(receipt.read_text())
        if r['status']!='PASS_PROBABILITY_SCREEN_MATURE_AUDITS':raise AssertionError('Audited control required')
        for row in r['rows']:
            if file_sha256(root/'runs'/cfg['phase']/f"PRIVATE-u{row['update']}.pt")!=row['checkpoint_sha256']:raise AssertionError('Control checkpoint changed')
        fit=root/'runs'/cfg['phase']/'report.json';f=json.loads(fit.read_text());initial_digests.add(f['initial_model_digest'])
        reused[name]=dict(audit_sha256=file_sha256(receipt),fit_report_sha256=file_sha256(fit),phase=cfg['phase'])
    if len(initial_digests)!=1:raise AssertionError('Previous initialization mismatch')
    import torch
    from compute_trainium.probability_model_factory import probability_model
    from scripts.qualify_native import digest
    torch.set_num_threads(1);torch.manual_seed(p['seed'])
    preflight=probability_model(dict(p,remaining_budget_conditioning=True,maximum_extra_steps=2))
    state=preflight.state_dict();shared=type(state)((n,v) for n,v in state.items() if n!='remaining_budget_cues')
    if digest(shared)!=next(iter(initial_digests)):raise AssertionError('Shared initialization must match before admission')
    if sum(v.numel() for v in preflight.parameters())!=62801408:raise AssertionError('Actual parameter inventory differs')
    del preflight,state,shared

    files={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('scripts/run_native_isolated.sh','docs/BUDGET_CONDITIONED_SCREEN_20261029.md'):files[f]=file_sha256(f)
    worker,host=split_snapshot_sources(files);names=[arm['phase'],ID+'-evaluation'];now=time.time()
    p.update(identity=ID,registered_unix=now,seconds_limit=7200,absolute_deadline_unix=now+7200,
        phase_seconds_limit=3600,cpu_phase_seconds_limit=1800,remaining_budget_conditioning=True,maximum_extra_steps=2,
        parameters=62801408,source_sha256=files,worker_source_sha256=worker,host_only_source_sha256=host,
        root_comm_ports=reserve_ports(root,names),phases={n:{} for n in names},inode_projection=projection,
        artifact_allowance_bytes=3*1024**3,qualification_report_sha256=file_sha256(gate),reused_arms=reused,
        expected_shared_initial_digest=initial_digests.pop(),parent_report_sha256=file_sha256(prior),
        parent_evaluation_identity=PARENT+'-evaluation',
        parent_score_sha256={f'PRIVATE-rank{rank}-scores.json':file_sha256(oldeval/f'PRIVATE-rank{rank}-scores.json') for rank in range(64)},
        scope='One sampled0/1/2 fit;only remaining-budget cues change versus prior sampled arm. Identical source order,shared initialization,optimizer,cooldown,matrix-work budget.1024 extra parameters explicitly counted;cue additions are elementwise and excluded under the unchanged matrix ledger. Existing audited controls reused without new fitting/scoring. Primary deepest-versus-ordinary-work and supported incremental steps remain mandatory;no learned allocation or automatic promotion.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def run(root):
    out=root/ID;e=ScreenExecution(root,out/'plan.json',out/'execution')
    r=dict(status='RUNNING_BUDGET_CUE_SCREEN',records={},new_fits=1)
    try:
        name=e.plan['arms']['workspace']['phase']
        r['records']['fit']=e.native(name,'scripts.train_probability_screen',
            ['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,
             '--arm','workspace','--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_PROBABILITY_SCREEN_FIT_PENDING_AUDITS')
        if r['records']['fit']['shared_initial_model_digest']!=e.plan['expected_shared_initial_digest']:raise AssertionError('Shared original weights changed')
        dest=out/'audit-workspace'
        r['records']['audit']=e.cpu(ID+'-audit','scripts.audit_probability_screen',
            ['--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--arm','workspace','--output',str(dest)],
            dest/'report.json','PASS_PROBABILITY_SCREEN_MATURE_AUDITS')
        write_state(PUBLIC/'progress.json',r);name=ID+'-evaluation'
        r['records']['evaluation']=e.native(name,'scripts.evaluate_probability_screen',
            ['--root','/work/instance-nvme','--plan',f'/work/instance-nvme/{ID}/plan.json','--plan-sha',e.sha,
             '--comm-port',str(e.plan['root_comm_ports'][name])],'COMPLETE_PROBABILITY_SCREEN_EVALUATION')
        dest=out/'analysis'
        r['records']['analysis']=e.cpu(ID+'-analysis','scripts.analyze_budget_workspace',
            ['--root',str(root),'--plan',str(out/'plan.json'),'--output',str(dest)],dest/'report.json','COMPLETE_BUDGET_CUE_ANALYSIS')
        write_state(PUBLIC/'analysis.json',r['records']['analysis']);r['status']='COMPLETE_BUDGET_CUE_SCREEN'
    except BaseException as error:r.update(status='STOPPED_BUDGET_CUE_SCREEN',error_type=type(error).__name__,error=str(error));raise
    finally:write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['prepare','run'],required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else run)(a.root)
