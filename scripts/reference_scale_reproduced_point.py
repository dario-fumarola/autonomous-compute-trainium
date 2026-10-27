"""Full512-window independent FP32 reference; no training/science/heldout readout."""
import argparse,json,time,sys
from pathlib import Path


def main(root,plan_file,out):
    started=time.time()
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from compute_trainium.training import file_sha256,atomic_checkpoint
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_training_candidate import map_after_shared_validation
    from compute_trainium.metrics import real_token_losses
    from compute_trainium.execution_isolation import managed_running
    from scripts.audit_parcae_precise_training_point import make_reference
    from scripts.qualify_parcae_candidate_cpu import PIN
    from scripts.qualify_native import digest
    import subprocess
    if out.exists():raise FileExistsError('Preserve every independent reference')
    out.mkdir();plan=json.loads(plan_file.read_text());code=Path(__file__).resolve().parents[1]
    if any(file_sha256(code/f)!=s for f,s in plan['source_sha256'].items()):raise AssertionError('Frozen diagnostic sources changed')
    if managed_running():raise RuntimeError('Independent CPU reference follows Native closure')
    upstream=root/'upstream-parcae-69284c1'
    if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):raise AssertionError('Independent pinned source changed')
    sys.path.insert(0,str(upstream));torch.set_num_threads(24)
    saved=torch.load(root/plan['point_file'],map_location='cpu',mmap=True,weights_only=False)
    if saved['eligible_for_science'] is not False:raise ValueError('Only ineligible reproduced point')
    cfg=ParcaeCandidateConfig(**plan['model']);model=make_reference(cfg,True);model.load_state_dict(saved['model'],strict=True);model.train();del saved
    before=digest(dict(model.state_dict()));packed=map_after_shared_validation(root/plan['data']['packed'],plan['data']['manifest_sha256'])
    cursor=plan['source_cursor'];count=sum(packed.counts(cursor,512));noises=[]
    for rank in range(64):
        file=root/plan['noise_directory']/f'rank{rank}.pt';a=json.loads((file.parent/f'rank{rank}-high.json').read_text());b=json.loads((file.parent/f'rank{rank}-low.json').read_text())
        if a!=b or file_sha256(file)!=a['sha256']:raise AssertionError('Both Native variants must use identical explicit noise')
        noises.append(torch.load(file,map_location='cpu',mmap=True,weights_only=False))
    current=None;model.initialize_state=lambda _:current;model.zero_grad(set_to_none=True);total=0.
    with sdpa_kernel(SDPBackend.MATH):
        for local in range(512):
            if time.time()>plan['absolute_deadline_unix']:raise TimeoutError('Original reproduced-point cap')
            available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            if available<544*1024**3:raise RuntimeError('Protected CPU reserve')
            rank,micro=local%64,local//64;window=cursor+local
            if noises[rank]['windows'][micro]!=window:raise AssertionError('Exact window/noise alignment')
            current=noises[rank]['states'][micro:micro+1];batch=packed.batch([window],0)[0]
            logits=model(batch.inputs,return_logits=True,num_steps_pair=torch.tensor([0,2]))['logits']
            loss=real_token_losses(logits,batch.targets,batch.target_mask).sum();(loss/count).backward();total+=float(loss.detach())
            if (local+1)%16==0:(out/'progress.json').write_text(json.dumps(dict(windows=local+1,total_windows=512,elapsed_seconds=time.time()-started))+'\n')
    gradients={n:p.grad for n,p in model.named_parameters()}
    finite=all(torch.isfinite(g).all() for g in gradients.values())
    norm=sum(float(g.detach().double().square().sum()) for g in gradients.values())**.5 if finite else None
    if finite:
        coefficient=min(1.,1./(norm+1e-6))
        for g in gradients.values():g.mul_(coefficient)
    after=digest(dict(model.state_dict()))
    if after!=before:raise AssertionError('Reference changed weights')
    atomic_checkpoint(out/'reference-gradient.pt',dict(gradients=gradients,cpu_nll=total/count,finite=finite,clipped=finite,
        raw_gradient_norm=norm,source_cursor=cursor,real_targets=count,windows=512,model_digest=before))
    result=dict(status='CAPTURED_INDEPENDENT_REPRODUCED_POINT_FP32_REFERENCE',windows=512,real_targets=count,cpu_nll=total/count,
        gradient_finite=finite,raw_gradient_norm=norm,norm_reduction_dtype='float64',model_unchanged=True,optimizer_updates=0,assigned_chips=0,
        gradient_sha256=file_sha256(out/'reference-gradient.pt'),wall_seconds=time.time()-started,scientific_fit_admitted=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.plan,a.output)
