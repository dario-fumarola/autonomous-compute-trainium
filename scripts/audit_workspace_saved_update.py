"""Independent CPU audit of the preserved Native update with FP64 global norm."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import torch

from compute_trainium.training import file_sha256
from compute_trainium.workspace_streaming_checkpoint import tensor_hash

PUBLIC=Path('results/workspace_full_update_cpu_audit_20261027')


def run(root):
    started=time.time();out=root/'workspace-full-update-cpu-audit-v1';parent=root/'runs/workspace-native-full-update-v1-gradient'
    if os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('Explicit independent CPU audit')
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every audit')
    native=json.loads((parent/'report.json').read_text())
    if native['status']!='PASS_WORKSPACE_FULL_MODEL_UPDATE_AND_RESTORE':raise AssertionError('Complete saved update/restore required')
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<536*1024**3 or shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Full CPU audit and protected reserve required')
        if time.time()-started>900:raise TimeoutError('Bounded independent full-state CPU audit')
    guard();torch.set_num_threads(12);out.mkdir();PUBLIC.mkdir()
    paths=[parent/f'rank{rank}-u1.zip' for rank in range(64)]
    plan=dict(registered_unix=time.time(),native_report_sha256=file_sha256(parent/'report.json'),
        native_shard_sha256=[file_sha256(p) for p in paths],optimizer_updates=0,assigned_chips=0,
        source_sha256={n:file_sha256(n) for n in ['scripts/audit_workspace_saved_update.py','compute_trainium/workspace_streaming_checkpoint.py']},
        scope='No Native state or history changes. Re-audit the same saved one-update result using the full independent HF gradient norm summed in FP64,not lower-precision display-family norms. Every actor/scorer/AdamW tensor slice is validated against its saved content hash before comparison.2% named delta/moment/EMA bound retained.')
    for d in [out,PUBLIC]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_WORKSPACE_SAVED_UPDATE_CPU_AUDIT',optimizer_updates=0,assigned_chips=0)
    archives=[]
    try:
        initial=torch.load(root/'workspace-reader-bootstrap-v1/checkpoint.pt',weights_only=True,mmap=True)['model']
        gradients=torch.load(root/'workspace-identity-gradient-v1/gradients.pt',weights_only=True,mmap=True)
        norm=sum(float(v.double().square().sum()) for v in gradients.values())**.5
        for p in paths:archives.append(zipfile.ZipFile(p))
        metadata=[torch.load(io.BytesIO(a.read('metadata.pt')),weights_only=True) for a in archives]
        names=metadata[0]['metadata']['parameter_names']
        if any(m['metadata']['parameter_names']!=names or m['metadata']['update']!=1 for m in metadata):raise AssertionError('Checkpoint counters/order differ')
        def read(rank,name):
            sig=metadata[rank]['signatures'][name]
            value=torch.load(io.BytesIO(archives[rank].read(sig['member'])),weights_only=True)
            if tensor_hash(value)!=sig['sha256']:raise AssertionError('Preserved tensor content changed')
            return value
        errors={label:{} for label in ['parameter_delta','adam_m','adam_v','ema_weights']}
        for index,name in enumerate(names):
            guard();(out/'progress.json').write_text(json.dumps(dict(parameter_index=index,parameters=len(names),wall_seconds=time.time()-started)))
            start=initial[name];p=torch.nn.Parameter(start.clone());p.grad=gradients[name].clone()*min(1.,1./(norm+1e-6))
            cpu=torch.optim.AdamW([p],lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False);cpu.step()
            totals={label:[0.,0.] for label in errors}
            for rank in range(64):
                c0=start.chunk(64,0)[rank];updated=p.detach().chunk(64,0)[rank]
                m=cpu.state[p]['exp_avg'].chunk(64,0)[rank];v=cpu.state[p]['exp_avg_sq'].chunk(64,0)[rank]
                for label,actual,expected in [
                    ('parameter_delta',read(rank,'model.'+name)-c0,updated-c0),
                    ('adam_m',read(rank,f'adam.{index}.exp_avg'),m),
                    ('adam_v',read(rank,f'adam.{index}.exp_avg_sq'),v),
                    ('ema_weights',read(rank,'scorer.'+name),c0*.5+updated*.5)]:
                    totals[label][0]+=float((actual.double()-expected.double()).square().sum())
                    totals[label][1]+=float(expected.double().square().sum())
                if float(read(rank,f'adam.{index}.step'))!=1.:raise AssertionError('AdamW step counter differs')
            for label,(diff,ref) in totals.items():errors[label][name]=(diff/max(ref,1e-30))**.5
            del p,cpu
        largest={k:max(v.values()) for k,v in errors.items()}
        if max(largest.values())>.02:raise AssertionError('Corrected complete CPU audit exceeds unchanged2% criterion')
        result.update(status='PASS_WORKSPACE_SAVED_UPDATE_CPU_AUDIT',global_fp64_reference_norm=norm,
            native_norm=native['gradient_norm'],largest_named_errors=largest,
            worst_named_parameters={k:max(v,key=v.get) for k,v in errors.items()},parameters_checked=len(names),
            tensor_slices_checked=len(names)*64*5,independent_complete_update_agreement=True,
            warning='One qualification update only. Native execution and historical receipts unchanged.No useful-writing or efficiency finding.')
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        for a in archives:a.close()
        result.update(wall_seconds=time.time()-started,plan_sha256=file_sha256(out/'plan.json'))
        for d in [out,PUBLIC]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
