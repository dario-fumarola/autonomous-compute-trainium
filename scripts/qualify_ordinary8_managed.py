"""Owned qualification bootstrap; no scientific training admission."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def validate_phase(plan,output,port,*,arm,mode):
    if (plan.get('purpose')!='ordinary8_native_qualification_only' or arm!='ordinary'
            or mode not in ['warm','replay'] or not 16000<=port<30000
            or plan['root_comm_ports'].get(output)!=port or plan['phases'][output]['mode']!=mode):
        raise ValueError('Exact owned ordinary8 qualification phase required')


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--qualification-plan',type=Path,required=True);p.add_argument('--qualification-sha',required=True)
    p.add_argument('--comm-port',type=int,required=True);p.add_argument('--resume-sha');a,remaining=p.parse_known_args()
    q=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--preparation-root',type=Path,required=True)
    q.add_argument('--arm',required=True);q.add_argument('--mode',required=True);q.add_argument('--resume',type=Path)
    q.add_argument('--accumulation',type=int,required=True);q.add_argument('--precision',required=True);b,_=q.parse_known_args(remaining)
    from compute_trainium.training import file_sha256
    if file_sha256(a.qualification_plan)!=a.qualification_sha:raise AssertionError('Numerical qualification plan changed')
    plan=json.loads(a.qualification_plan.read_text());validate_phase(plan,b.output.name,a.comm_port,arm=b.arm,mode=b.mode)
    if (b.accumulation!=16 or b.precision!='fp16' or b.preparation_root.resolve()!=
            (a.qualification_plan.parent.parent/plan['preparation_directory']).resolve()
            or file_sha256(b.preparation_root/'plan.json')!=plan['preparation_plan_sha256']):raise AssertionError('Exact prepared geometry/precision required')
    repo=Path(__file__).resolve().parents[1]
    if any(file_sha256(repo/f)!=sha for f,sha in plan['worker_source_sha256'].items()):raise AssertionError('Exact worker snapshot changed')
    if b.mode=='replay':
        parent=plan['phases'][b.output.name]['parent'];expected=a.qualification_plan.parent.parent/'runs'/parent/'after-120-full-state.pt'
        if b.resume is None or b.resume.resolve()!=expected.resolve() or not a.resume_sha or file_sha256(b.resume)!=a.resume_sha:
            raise AssertionError('Checksum-pinned warm checkpoint required')
    elif b.resume is not None or a.resume_sha is not None:raise ValueError('Warm qualification starts only at C0')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
    from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer,changing_qualification_schedule
    from scripts import train_parcae_precise_qualification as engine
    engine.ParcaeFP16Candidate=ParcaePrimitiveSDPACandidate;engine.ParcaeOptimizerCandidate=ParcaeRuntimeScalarOptimizer
    engine.configure_optimizer=changing_qualification_schedule;sys.argv=[sys.argv[0],*remaining];engine.main()
    if int(os.environ.get('RANK','0'))==0:
        file=b.output/'report.json';row=json.loads(file.read_text());row.update(managed_qualification_plan_sha256=a.qualification_sha,
            execution_variant='ordinary8_fp16_primitive_attention_runtime_scalars');file.write_text(json.dumps(row,indent=2)+'\n')


if __name__=='__main__':main()
