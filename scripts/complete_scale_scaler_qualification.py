"""Bounded qualification closure chain; never launches a scientific fit."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path

ROOT=Path('/run/user/1000/autonomous-compute/instance-nvme')
CODE=Path(__file__).resolve().parents[1]
CHAIN=ROOT/'scale-65536-qualification-chain-v1'
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=json.loads((CHAIN/'plan.json').read_text());deadline=plan['absolute_deadline_unix'];children=[]
    def guard():
        if time.time()>deadline or SHA(Path(__file__))!=plan['source_sha256']:raise TimeoutError('Original bounded qualification-chain clock/source')
    def run(command,log):
        guard()
        with log.open('x') as stream:
            p=subprocess.Popen(command,cwd=CODE,env={**os.environ,'USE_TORCH_XLA':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'24','MKL_NUM_THREADS':'24','PYTHONPATH':str(CODE)},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p)
        while p.poll() is None:guard();time.sleep(2)
        if p.returncode:raise RuntimeError('Qualification phase failed; preserved log '+log.name)
        return p.pid
    result=dict(status='STOPPED_HIGHER_SCALER_QUALIFICATION_CHAIN',records={},scientific_fit_admitted=False)
    try:
        first=ROOT/'scale-65536-fixed-two-numerical-v1'
        while not (first/'report.json').exists():
            guard()
            if subprocess.run(['kill','-0',str(plan['recurrent_controller_pid'])],capture_output=True).returncode:raise RuntimeError('Original recurrent controller stopped without a receipt')
            time.sleep(2)
        for role,label in [('fixed_two','fixed-two'),('ordinary8','ordinary')]:
            identity='scale-65536-'+label+'-numerical-v1';native=ROOT/identity
            if role=='ordinary8':
                run(['/home/ubuntu/ema-trainium/.venv/bin/python','-B','-m','scripts.run_scale_scaler_qualification','--root',str(ROOT),'--role',role],CHAIN/'ordinary-prepare.log')
                run(['/home/ubuntu/ema-trainium/.venv/bin/python','-B','-m','scripts.run_scale_scaler_qualification','--root',str(ROOT),'--role',role,'--run'],CHAIN/'ordinary-native.log')
            capture=json.loads((native/'report.json').read_text())
            if capture['status']!='CAPTURED_SCALE_ACTUAL_SHAPE_CPU_AUDITS_PENDING':raise RuntimeError('Native qualification did not capture completely')
            cpu=ROOT/('scale-65536-'+label+'-independent-cpu-v1');cpu.mkdir();start=time.time()
            native_plan=json.loads((native/'plan.json').read_text())
            registration=dict(identity=cpu.name,native_identity=identity,purpose='independent_scale_numerical_cpu_only',
                registered_unix=start,absolute_deadline_unix=min(start+7200,deadline),source_sha256=native_plan['source_sha256'],
                native_plan_sha256=SHA(native/'plan.json'),native_capture_report_sha256=SHA(native/'report.json'),
                maximum_cpu_optimizer_steps=1,maximum_global_objective_windows=512,assigned_chips=0,scientific_fit_admitted=False)
            (cpu/'plan.json').write_text(json.dumps(registration,indent=2)+'\n')
            pid=run(['/home/ubuntu/ema-trainium/.venv/bin/python','-B','-m','scripts.audit_scale_model','--root',str(ROOT),'--output',str(cpu/'audit'),'--registration',str(cpu/'plan.json')],cpu/'controller.log')
            audit=json.loads((cpu/'audit/report.json').read_text())
            if (audit['status']!='PASS_SCALE_ACTUAL_SHAPE_INDEPENDENT_CPU_AUDITS' or not audit['first_update']['passes'] or not audit['full512']['gradient']['passes_two_percent_gate'] or not audit['replay']['passes']):raise RuntimeError('Complete original numerical gates required')
            proof=dict(status='PASS_SCALE_65536_COMPLETE_NUMERICAL_QUALIFICATION',role=role,manual_loss_scale=65536,
                native_plan_sha256=SHA(native/'plan.json'),independent_cpu_report_sha256=SHA(cpu/'audit/report.json'),
                full512_gradient=audit['full512']['gradient'],native_minus_cpu_nll=audit['full512']['native_minus_cpu_nll'],
                first_update_moments_pass=True,replay=audit['replay'],actual_qualification_updates=102,scientific_fit_admitted=False)
            (native/'completion-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
            public=CODE/'results/scale_65536_numerical_20261027'/role
            (public/'completion-verification.json').write_text(json.dumps(proof,indent=2)+'\n');(public/'independent-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
            result['records'][role]=proof;(CHAIN/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        result['status']='PASS_BOTH_HIGHER_SCALER_NUMERICAL_QUALIFICATIONS'
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
        result['finished_unix']=time.time();(CHAIN/'report.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
