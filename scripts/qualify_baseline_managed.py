"""Three explicit baseline constructors around the existing numerical trainer."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def validate_phase(plan,output,port,*,profile,arm,mode):
    if plan.get('purpose')!='baseline_native_qualification_only':raise ValueError('Numerical-only baseline plan required')
    expected={'one':'fixed1','mixing':'poisson2','ordinary8':'ordinary'}
    if (profile not in expected or arm!=expected[profile] or mode not in ['warm','replay']
            or not 16000<=port<30000 or plan['root_comm_ports'].get(output)!=port
            or plan['phases'][output]['profile']!=profile or plan['phases'][output]['mode']!=mode):
        raise ValueError('Exact owned profile,depth and numerical phase required')


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--qualification-plan',type=Path,required=True);p.add_argument('--qualification-sha',required=True)
    p.add_argument('--comm-port',type=int,required=True);p.add_argument('--profile',required=True);p.add_argument('--resume-sha')
    a,remaining=p.parse_known_args()
    q=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--preparation-root',type=Path,required=True)
    q.add_argument('--arm',required=True);q.add_argument('--mode',required=True);q.add_argument('--resume',type=Path)
    q.add_argument('--accumulation',type=int,required=True);q.add_argument('--precision',required=True);b,_=q.parse_known_args(remaining)
    from compute_trainium.training import file_sha256
    if file_sha256(a.qualification_plan)!=a.qualification_sha:raise AssertionError('Frozen baseline qualification changed')
    plan=json.loads(a.qualification_plan.read_text());validate_phase(plan,b.output.name,a.comm_port,profile=a.profile,arm=b.arm,mode=b.mode)
    prep=plan['profiles'][a.profile]
    if (b.accumulation!=16 or b.precision!='fp16' or b.preparation_root.resolve()!=
            (a.qualification_plan.parent.parent/prep['preparation_directory']).resolve()
            or file_sha256(b.preparation_root/'plan.json')!=prep['preparation_plan_sha256']):raise AssertionError('Exact prepared numerical geometry required')
    repo=Path(__file__).resolve().parents[1]
    if any(file_sha256(repo/f)!=sha for f,sha in plan['worker_source_sha256'].items()):raise AssertionError('Complete numerical source snapshot changed')
    if b.mode=='replay':
        parent=plan['phases'][b.output.name]['parent'];expected=a.qualification_plan.parent.parent/'runs'/parent/'after-120-full-state.pt'
        if b.resume is None or b.resume.resolve()!=expected.resolve() or not a.resume_sha or file_sha256(b.resume)!=a.resume_sha:
            raise AssertionError('Exact complete warm-state replay required')
    elif b.resume is not None or a.resume_sha is not None:raise ValueError('Qualification must start only at original C0')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
    from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer,changing_qualification_schedule
    from scripts import train_parcae_precise_qualification as engine
    engine.ParcaeFP16Candidate=ParcaePrimitiveSDPACandidate;engine.ParcaeOptimizerCandidate=ParcaeRuntimeScalarOptimizer
    engine.configure_optimizer=changing_qualification_schedule
    if a.profile=='mixing':
        from compute_trainium.lean_mixing_precision import LeanFP32MixingCandidate
        from compute_trainium.parcae_mixing_control import mixing_only_ledger
        engine.ParcaeFP16Candidate=LeanFP32MixingCandidate
        engine.ParcaeStagesCandidate=lambda *args,**kw:LeanFP32MixingCandidate(*args,fp16_body=False,**kw)
        engine.model_work=lambda c,depths,context=1024:sum(mixing_only_ledger(c,context,total_steps=int(n),backward_cap=int(k))['training_matmul_flops'] for n,k in depths)
    sys.argv=[sys.argv[0],*remaining];engine.main()
    if int(os.environ.get('RANK','0'))==0:
        file=b.output/'report.json';row=json.loads(file.read_text());row.update(managed_qualification_plan_sha256=a.qualification_sha,
            qualification_profile=a.profile);file.write_text(json.dumps(row,indent=2)+'\n')


if __name__=='__main__':main()
