"""Owned-port bootstrap around the unchanged prepared lean constructor."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def validate_phase(plan, output, port, *, arm, mode):
    if plan.get('purpose')!='lean_ntp_native_qualification_only':
        raise ValueError('Qualification-only owned execution required')
    if not 16000<=port<30000 or plan['root_comm_ports'].get(output)!=port:
        raise ValueError('Exact registered owned phase/port required')
    phase=plan['phases'][output]
    if phase['arm']!=arm or phase['mode']!=mode:
        raise ValueError('Phase cannot change the declared model or replay mode')


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--qualification-plan',type=Path,required=True)
    p.add_argument('--qualification-sha',required=True);p.add_argument('--comm-port',type=int,required=True)
    p.add_argument('--resume-sha');a,remaining=p.parse_known_args()
    q=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--preparation-root',type=Path,required=True)
    q.add_argument('--arm',required=True);q.add_argument('--mode',required=True);q.add_argument('--resume',type=Path)
    b,_=q.parse_known_args(remaining)
    from compute_trainium.training import file_sha256
    if file_sha256(a.qualification_plan)!=a.qualification_sha:raise AssertionError('Qualification plan changed')
    plan=json.loads(a.qualification_plan.read_text())
    validate_phase(plan,b.output.name,a.comm_port,arm=b.arm,mode=b.mode)
    expected=a.qualification_plan.parent.parent/plan['preparation_directory']
    if b.preparation_root.resolve()!=expected.resolve() or file_sha256(b.preparation_root/'plan.json')!=plan['preparation_plan_sha256']:
        raise AssertionError('Prepared lean inputs changed')
    code_root=Path(__file__).resolve().parents[1]
    from compute_trainium.native_snapshot_manifest import split_snapshot_sources
    expected_sources,_=split_snapshot_sources(plan['source_sha256'])
    if (expected_sources!=plan['worker_source_sha256']
            or 'scripts/qualify_lean_ntp_managed.py' not in expected_sources
            or 'scripts/train_parcae_precise_qualification.py' not in expected_sources):
        raise AssertionError('Complete explicit managed worker manifest required')
    if any(file_sha256(code_root/file)!=sha for file,sha in plan['worker_source_sha256'].items()):
        raise AssertionError('Exact owned executable snapshot changed')
    if b.mode=='replay':
        parent=plan['phases'][b.output.name]['parent']
        checkpoint=a.qualification_plan.parent.parent/'runs'/parent/'after-120-full-state.pt'
        if b.resume is None or b.resume.resolve()!=checkpoint.resolve() or not a.resume_sha or file_sha256(b.resume)!=a.resume_sha:
            raise AssertionError('Exact registered warm-state replay required')
    elif b.resume is not None or a.resume_sha is not None:
        raise ValueError('Warm phase starts only from qualification initialization')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    from scripts import qualify_lean_ntp_primitives as prepared
    sys.argv=[sys.argv[0],*remaining];prepared.main()
    if int(os.environ.get('RANK','0'))==0:
        file=b.output/'report.json';row=json.loads(file.read_text())
        row.update(managed_qualification_plan_sha256=a.qualification_sha,
                   execution_variant='lean_fp16_primitive_attention_runtime_scalars',
                   qualification_schedule='existing_changing_qualification_schedule_peak_1e-4')
        file.write_text(json.dumps(row,indent=2)+'\n')


if __name__=='__main__':main()
