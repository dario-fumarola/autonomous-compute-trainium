"""Explicit readout-only recovery under the unchanged original TUNE clock."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path

from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution

IDENTITY='scale-tune-readout-recovery-v1'
PUBLIC=Path('results/scale_tune_readout_recovery_20261027')
ADDED=['scripts/resume_scale_tune_readout.py','scripts/score_scale_readout_recovery.py','compute_trainium/canonical_prediction_checkpoint.py']


def prepare(root):
    import torch
    from scripts.qualify_native import digest
    folder=root/IDENTITY;original=root/'scale-medium-tune-65536-v1'
    if folder.exists() or PUBLIC.exists():raise FileExistsError('Preserve every readout recovery')
    plan_file=original/'plan.json';plan=json.loads(plan_file.read_text());stopped=json.loads((original/'report.json').read_text())
    if stopped['status']!='STOPPED_SCALE_TUNE' or stopped.get('error')!='CPU phase failed: fixed-two-lr0-score':raise ValueError('Exact stopped readout required')
    if any(file_sha256(f)!=s for f,s in plan['source_sha256'].items()):raise AssertionError('Every original listed scientific source must remain unchanged')
    if list((original/'scores').glob('*/report.json')):raise ValueError('No quality score may precede this correction')
    first=plan['cases']['fixed-two-lr0'];score=root/first['score_directory']
    if list(score.iterdir()):raise ValueError('Only the original empty failed score directory may move')
    retained=root/first['retained_model'];saved=torch.load(retained,map_location='cpu',mmap=True,weights_only=False)
    expected=saved['model_digest'];canonical=digest(dict(saved['model']));raw=digest(saved['model'])
    if canonical!=expected or raw==expected:raise AssertionError('Exact OrderedDict-versus-dict diagnosis required')
    del saved
    if time.time()>=plan['absolute_deadline_unix']:raise TimeoutError('Original TUNE clock expired')
    folder.mkdir();PUBLIC.mkdir();preserved=root.parent/'recurrence'/IDENTITY;preserved.mkdir(exist_ok=False)
    old=preserved/'original-ordered-readout.pt';sha=file_sha256(retained);retained.rename(old)
    failed_score=original/'scores/fixed-two-lr0-before-readout-recovery';score.rename(failed_score)
    recovery=dict(identity=IDENTITY,registered_unix=time.time(),original_study='scale-medium-tune-65536-v1',
        original_plan_sha256=file_sha256(plan_file),original_stopped_report_sha256=file_sha256(original/'report.json'),
        absolute_deadline_unix=plan['absolute_deadline_unix'],original_scale_deadline_unix=plan['scale_budget_deadline_unix'],
        additional_source_sha256={f:file_sha256(f) for f in ADDED},original_scientific_sources_unchanged=True,
        prior_complete_mature_audit_sha256=file_sha256(root/first['audit_directory']/'report.json'),
        replacement_retained_path=str(retained),preserved_retained_path=str(old),preserved_retained_sha256=sha,
        original_readback_digest=raw,canonical_readback_digest=canonical,expected_pre_save_digest=expected,
        extra_optimizer_updates=0,original_rates_and_cases_unchanged=True,
        scope='Container canonicalization only at retained prediction serialization; every tensor/name/dtype/shape retains the pre-save digest. Original failed output and empty failed score directory remain preserved. All original sources,model weights,targets,CPU arithmetic,causal noise,cases,selection and clocks remain fixed. Successful original Native/audit phases reuse their exact completed receipts; partial phases cannot restart. No automatic main.')
    write_state(folder/'plan.json',recovery)
    public={k:v for k,v in recovery.items() if k not in ['replacement_retained_path','preserved_retained_path']};write_state(PUBLIC/'plan.json',public)


def run(root):
    folder=root/IDENTITY;registration=folder/'plan.json';recovery=json.loads(registration.read_text());sha=file_sha256(registration)
    original=root/recovery['original_study'];plan_file=original/'plan.json'
    if file_sha256(plan_file)!=recovery['original_plan_sha256']:raise AssertionError('Original study changed')
    execution=AllChipExecution(root,plan_file,original/'execution',resume=True);records={}
    result=dict(status='STOPPED_SCALE_READOUT_RECOVERY',records=records,scientific_main_fit_admitted=False)
    def guard():
        execution.check()
        if file_sha256(registration)!=sha or any(file_sha256(f)!=s for f,s in recovery['additional_source_sha256'].items()):raise AssertionError('Frozen readout-only recovery changed')
    def stop(number,frame):raise KeyboardInterrupt('Original bounded recovery interrupted')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        for key in execution.plan['case_order']:
            guard();case=execution.plan['cases'][key];record={};records[key]=record
            for suffix,case_key in [('warm',key),('replay',key+'-replay')]:
                name=recovery['original_study']+'-'+key+'-'+suffix;file=root/'runs'/name/'report.json'
                if file.exists():
                    value=json.loads(file.read_text());closure=json.loads((root/'snapshots'/name/'closure.json').read_text())
                    if value['plan_sha256']!=execution.sha or value['status']!='CAPTURED_SCALE_REGISTERED_TRAINING_PHASE' or value['world']!=64 or closure['exit_code']!=0:raise ValueError('Only exact successful original phase may be reused')
                    record[suffix]=value
                else:
                    options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+recovery['original_study']+'/plan.json','--plan-sha',execution.sha,'--case',case_key,'--comm-port',str(execution.plan['root_comm_ports'][name])]
                    record[suffix]=execution.native(name,'scripts.train_scale_registered',options,'CAPTURED_SCALE_REGISTERED_TRAINING_PHASE')
                write_state(folder/'progress.json',result)
            record['audit']=execution.cpu(key+'-audit','scripts.audit_scale_registered',
                ['--root',str(root),'--plan',str(plan_file),'--case',key,'--output',str(root/case['audit_directory'])],
                root/case['audit_directory']/'report.json','PASS_SCALE_REGISTERED_ENDPOINT_NUMERICAL_AUDIT')
            guard()
            record['score']=execution.cpu(key+'-score-recovered','scripts.score_scale_readout_recovery',
                ['--readout-recovery',str(registration),'--readout-recovery-sha',sha,
                 '--root',str(root),'--plan',str(plan_file),'--case',key,'--output',str(root/case['score_directory'])],
                root/case['score_directory']/'report.json','COMPLETE_SCALE_CONSUMED_DEVELOPMENT_READOUT')
            guard()
            cleanup=subprocess.run(['sudo','-n','env','USE_TORCH_XLA=0','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(Path('.').resolve()),os.sys.executable,'-B','-m','scripts.close_scale_tune_diagnostics','--root',str(root),'--plan',str(plan_file),'--case',key],capture_output=True,text=True,timeout=300)
            if cleanup.returncode:raise RuntimeError('Original successful-only diagnostic lifecycle failed: '+cleanup.stderr[-2000:])
            record['cleanup']=json.loads((root/case['cleanup_receipt']).read_text());write_state(folder/'progress.json',result)
        selected={}
        for role in ['fixed_two','ordinary8']:
            keys=[k for k in execution.plan['case_order'] if execution.plan['cases'][k]['model_role']==role]
            eligible=[k for k in keys if records[k]['score']['metrics']['weighted_nll']<execution.plan['selection']['healthy_maximum_nll']]
            if not eligible:raise RuntimeError('No eligible healthy independently tuned role')
            key=min(eligible,key=lambda k:(records[k]['score']['metrics']['weighted_nll'],execution.plan['cases'][k]['peak_lr']))
            selected[role]=dict(case=key,rate=execution.plan['cases'][key]['peak_lr'],score=records[key]['score']['metrics']['weighted_nll'],boundary_optimum=execution.plan['cases'][key]['peak_lr'] in [.008,.064])
        result.update(status='COMPLETE_INDEPENDENT_SCALE_RATE_SELECTION',selected=selected)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(plan_sha256=execution.sha,readout_recovery_sha256=sha,original_stopped_report_sha256=recovery['original_stopped_report_sha256'])
        execution.close(folder/'execution-costs.json');write_state(folder/'report.json',result);write_state(PUBLIC/'report.json',result)
        for s,handler in handlers.items():signal.signal(s,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    run(a.root) if a.run else prepare(a.root)
