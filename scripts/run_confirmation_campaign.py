"""Persistent bounded follow-up, independent audits and twelve-model TEST release."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from compute_trainium.confirmation import endpoint_inventory
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--absolute-deadline-unix',type=float,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);(a.output/'queue').mkdir()
    root=Path(os.environ['RESEARCH_ROOT']);croot=os.environ['RESEARCH_CONTAINER_ROOT'];started=time.time()
    state={'stages':{'original_science':{'status':'RUNNING'}}}
    deadline=a.absolute_deadline_unix
    load=lambda p:json.loads(p.read_text())
    def save():state['heartbeat_unix']=time.time();write_state(a.output/'queue/state.json',state)
    def done(status,**details):
        save();write_state(a.output/'queue/report.json',{'status':status,**details,'elapsed_seconds':time.time()-started})
    def wait(path,process=None,worker_output=None):
        while not path.exists():
            if time.time()>deadline:raise TimeoutError('Declared confirmation deadline')
            if process is not None and process.poll() is not None:
                raise RuntimeError('Prerequisite worker exited without its result: '+str(path))
            if worker_output is not None:
                running=False
                for entry in Path('/proc').glob('[0-9]*/cmdline'):
                    try:
                        command=entry.read_bytes().split(b'\0')
                        if b'scripts.verify_mechanism_cpu' in command and os.fsencode(worker_output) in command:running=True;break
                    except (FileNotFoundError,PermissionError):pass
                if not running:raise RuntimeError('Original CPU audit worker exited without a result')
            save();time.sleep(5)
        return load(path)
    def stage(name,status):state['stages'][name]={'status':status};save()
    def background(module,arguments,label):
        log=(a.output/(label+'.log')).open('w')
        proc=subprocess.Popen([os.sys.executable,'-u','-m',module,*arguments],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();write_state(a.output/(label+'-process.json'),{'pid':proc.pid,'started_unix':time.time(),'module':module})
        return proc
    def execute(run,model,split,calibration=False):
        if time.time()+150>deadline:raise TimeoutError('Trained-path phase budget exhausted')
        subprocess.run(['docker','update','--memory',str(224*1024**3),'autonomous-compute-neuron233'],check=True,capture_output=True)
        env=dict(os.environ,RESEARCH_RUN=run,RESEARCH_ENTRY_MODULE='scripts.evaluate_trained_path',RESEARCH_DDP_RANKS='64',
            NEURON_RT_VISIBLE_CORES=','.join(map(str,range(64))),RESEARCH_TIMEOUT_SECONDS='900')
        corpus='/work/natural-v2-1050m/dev.json' if split=='dev' else '/work/natural-v2-1050m/test-sealed.json'
        command=[os.sys.executable,'-m','scripts.supervise','--checkpoint',model['checkpoint'],'--corpus',corpus,
            '--repetitions',str(model['passes']),'--split',split]
        if calibration:command+=['--calibration-report',model['reference']]
        if split=='test':command+=['--test-release',f'{croot}/{a.output.relative_to(root)}/test-release.json']
        subprocess.run(command,env=env,check=True)
        report=load(root/'runs'/run/'report.json')
        if report['status']!='PASS_SHARDED_TRAINED_PATH_READOUT':raise ValueError('Trained-path readout failed')
        return report
    save()
    try:
        original=wait(a.campaign/'queue/report.json')
        if original['status']!='COMPLETE_QUEUE':done('CONFIRMATION_BLOCKED_ORIGINAL_CAMPAIGN',prior_status=original['status']);return
        stage('original_science','COMPLETE')
        deadline=min(a.absolute_deadline_unix,time.time()+4*3600)
        write_state(a.output/'plan.json',{'identity':'confirmation-campaign-v3','deadline_unix':deadline,
            'maximum_followup_seconds':4*3600,'absolute_deadline_unix':a.absolute_deadline_unix,
            'protocol_sha256':file_sha256('docs/CONFIRMATION_FOLLOWUP_20261023.md'),
            'scope':'Fixed untied seeds47/71, one annealed estimated-compute endpoint, independent audits, trained-path TEST. No new architecture.'})
        background('scripts.campaign_deadline',['--campaign',str(a.output)],'deadline-guard')
        subprocess.run([os.sys.executable,'-m','scripts.analyze_sharing_exposure','--root',str(root),'--output','results/sharing_exposure_20261023'],check=True)
        stage('dev_template_calibration','RUNNING');models=endpoint_inventory(croot)
        for arm in ['ordinary','shared','untied']:
            model=next(m for m in models if m['arm']==arm and m['seed']==23 and m['endpoint']=='1B')
            execute('confirmation-v3-calibration-'+arm,model,'dev',True)
        stage('dev_template_calibration','COMPLETE')
        stage('original_independent_audits','RUNNING')
        if wait(root/'mechanism-endpoint-cpu-v1/report.json',worker_output=root/'mechanism-endpoint-cpu-v1')['status']!='PASS_THREE_NEW_ENDPOINT_CPU_AUDITS':
            raise ValueError('Original independent numerical audit failed')
        stage('original_independent_audits','COMPLETE')
        # All identities are fixed before their fit. Numerical audits overlap fits.
        audit_models=[['ordinary','confirmation-v3-ordinary-flop-s23-u6496'],
            ['untied','confirmation-v3-untied-s47-u3840'],['untied','confirmation-v3-untied-s71-u3840']]
        write_state(a.output/'audit-models.json',audit_models)
        audit_process=background('scripts.verify_mechanism_cpu',['--root',str(root),'--output',str(a.output/'independent-cpu'),
            '--deadline-unix',str(deadline-600),'--models-manifest',str(a.output/'audit-models.json')],'independent-cpu')
        write_state(a.output/'recovery-models.json',[run for _,run in audit_models])
        copy_process=background('scripts.preserve_mechanism_checkpoints',['--root',str(root),
            '--recovery','/home/ubuntu/autonomous-compute-recovery','--deadline-unix',str(deadline),
            '--runs-manifest',str(a.output/'recovery-models.json'),'--output',str(a.output/'checkpoint-copy-receipt.json')],'checkpoint-copy')
        stage('followup_fits','RUNNING')
        # No RESEARCH_ENTRY_MODULE override leaks into run_phases' train/eval routing.
        env=dict(os.environ);env.pop('RESEARCH_ENTRY_MODULE',None)
        subprocess.run([os.sys.executable,'-u','-m','scripts.run_confirmation_followups','--output',str(a.output/'followups'),
            '--deadline-unix',str(deadline)],env=env,check=True)
        follow=load(a.output/'followups/report.json')
        if follow['status']!='COMPLETE_CONFIRMATION_FOLLOWUPS':raise ValueError('Follow-up models incomplete')
        if follow['specification']['flop_match']['updates']!=6496:raise ValueError('Predeclared integer endpoint differs')
        stage('followup_fits','COMPLETE')
        subprocess.run([os.sys.executable,'-m','scripts.analyze_confirmation_followups','--root',str(root),
            '--output','results/confirmation_followup_20261023'],check=True)
        stage('followup_independent_audits','RUNNING')
        if wait(a.output/'independent-cpu/report.json',process=audit_process)['status']!='PASS_THREE_NEW_ENDPOINT_CPU_AUDITS':
            raise ValueError('Follow-up independent numerical audit failed')
        stage('followup_independent_audits','COMPLETE')
        if time.time()+1800>deadline:raise TimeoutError('Protected complete TEST budget no longer fits')
        source_files=['scripts/evaluate_trained_path.py','scripts/analyze_frozen_test.py','scripts/analyze_confirmation_followups.py',
            'compute_trainium/confirmation.py','compute_trainium/evaluation_gate.py','compute_trainium/metrics.py',
            'compute_trainium/training.py','compute_trainium/data.py','compute_trainium/titan.py','compute_trainium/recurrence.py',
            'compute_trainium/preln_recurrence.py','compute_trainium/model.py']
        source={f:file_sha256(f) for f in source_files};allowed=[]
        for m in models:
            cp=root.parent/Path(m['checkpoint']).relative_to('/work')
            if not cp.exists():raise ValueError('Predeclared terminal checkpoint missing')
            sha=file_sha256(cp);allowed.append(sha+':'+str(m['passes']))
            m.update(checkpoint_sha256=sha,run='confirmation-v3-test-'+m['identity'])
        release={'status':'FROZEN_TEST_RELEASE','frozen_unix':time.time(),'models':models,
            'corpus_sha256':file_sha256(root.parent/'natural-v2-1050m/test-sealed.json'),
            'allowed_checkpoint_unrolls':allowed,'evaluator_sha256':source['scripts/evaluate_trained_path.py'],
            'analysis_sha256':source['scripts/analyze_frozen_test.py'],'source_sha256':source,
            'test_protocol_sha256':file_sha256('docs/FROZEN_TEST_CONFIRMATION.md'),
            'followup_protocol_sha256':file_sha256('docs/CONFIRMATION_FOLLOWUP_20261023.md')}
        write_state(a.output/'test-release.json',release);stage('frozen_test','RUNNING')
        for m in models:
            for f,sha in source.items():
                if file_sha256(f)!=sha:raise ValueError('Frozen TEST code or analysis changed')
            execute(m['run'],m,'test');state['stages']['frozen_test']['last_completed']=m['identity'];save()
        stage('frozen_test','COMPLETE')
        subprocess.run([os.sys.executable,'-m','scripts.analyze_frozen_test','--root',str(root),'--campaign',str(a.output),
            '--output','results/frozen_test_20261023'],check=True)
        stage('full_state_recovery_copies','RUNNING')
        if wait(a.output/'checkpoint-copy-receipt.json',process=copy_process)['status']!='PASS_FULL_STATE_EBS_RECOVERY_COPIES':
            raise ValueError('Private full-state recovery copies incomplete')
        stage('full_state_recovery_copies','COMPLETE')
        done('COMPLETE_MECHANISM_POSTPROCESSING')
    except BaseException as error:
        done('CONFIRMATION_CAMPAIGN_FAILED',type=type(error).__name__,error=str(error),
            test_release_created=(a.output/'test-release.json').exists())
        raise


if __name__=='__main__':main()
