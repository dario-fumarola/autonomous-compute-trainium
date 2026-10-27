"""Bounded zero-update CPU reference timing/parity on one authentic TRAIN window."""
import argparse,json,statistics,time
from pathlib import Path


def main(root,out):
    started=time.time();deadline=started+300
    if out.exists():raise FileExistsError('Preserve every CPU timing attempt')
    out.mkdir(parents=True)
    import torch,sys
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_training_candidate import initial_state,gradient_comparison,map_after_shared_validation
    from compute_trainium.metrics import real_token_losses
    from compute_trainium.training import file_sha256
    from scripts.audit_parcae_precise_training_point import make_reference
    from scripts.qualify_native import digest
    original=root/'scale-medium-tune-65536-v1';plan=json.loads((original/'plan.json').read_text());case=plan['cases']['fixed-two-lr0']
    source=root/'upstream-parcae-69284c1';sys.path.insert(0,str(source))
    torch.set_num_threads(24)
    saved=torch.load(root/case['retained_model'],map_location='cpu',mmap=True,weights_only=False)
    if digest(dict(saved['model']))!=saved['model_digest']:raise AssertionError('Exact retained model required')
    config=ParcaeCandidateConfig(**case['model']);model=make_reference(config,True);model.load_state_dict(saved['model'],strict=True);model.train();del saved
    before=digest(dict(model.state_dict()));packed=map_after_shared_validation(root/plan['data']['packed'],plan['data']['manifest_sha256'])
    batch=packed.batch([0],0)[0];model.initialize_state=lambda _:initial_state(config,0,case['state_seed'])
    inputs=batch.inputs;targets=batch.targets;mask=batch.target_mask
    plan_receipt=dict(started_unix=started,absolute_deadline_unix=deadline,thread_order=[24,48,96,12],
        source_window=0,source_manifest_sha256=plan['data']['manifest_sha256'],model_digest=before,
        warm_iterations=1,timed_iterations=2,maximum_forward_backward_calls=12,assigned_chips=0,optimizer_updates=0,
        research_examples_generated=0,source_and_targets_unchanged=True)
    (out/'plan.json').write_text(json.dumps(plan_receipt,indent=2)+'\n');reference=None;reference_nll=None;records=[]
    with sdpa_kernel(SDPBackend.MATH):
        for threads in plan_receipt['thread_order']:
            torch.set_num_threads(threads);times=[]
            for iteration in range(3):
                if time.time()>deadline:raise TimeoutError('Original five-minute CPU performance cap')
                available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
                if available<544*1024**3:raise RuntimeError('Protected independent CPU reference reserve')
                model.zero_grad(set_to_none=True);tick=time.perf_counter()
                logits=model(inputs,return_logits=True,num_steps_pair=torch.tensor([0,2]))['logits']
                loss=real_token_losses(logits,targets,mask).sum()/mask.sum();loss.backward()
                elapsed=time.perf_counter()-tick
                if iteration:times.append(elapsed)
            gradients={n:p.grad for n,p in model.named_parameters()}
            if reference is None:reference={n:g.clone() for n,g in gradients.items()};reference_nll=float(loss.detach())
            comparison=gradient_comparison(reference,gradients)
            if not comparison['passes_two_percent_gate'] or abs(float(loss.detach())-reference_nll)>.003:raise AssertionError('CPU execution-thread parity failed')
            records.append(dict(threads=threads,timed_forward_backward_seconds=times,median_seconds=statistics.median(times),
                nll_difference_from24=float(loss.detach())-reference_nll,gradient=comparison))
            (out/'progress.json').write_text(json.dumps(dict(records=records),indent=2)+'\n')
    if digest(dict(model.state_dict()))!=before:raise AssertionError('Zero-update CPU measurement changed model')
    result=dict(status='COMPLETE_CPU_REFERENCE_THREAD_TIMING_PARITY',records=records,wall_seconds=time.time()-started,
        model_unchanged=True,assigned_chips=0,optimizer_updates=0,research_examples_generated=0,
        diagnostic_input_presentations=12,unique_source_windows=1,
        source_sha256=file_sha256(Path(__file__)),scope='Single authentic TRAIN window,unchanged trained weights,full FP32 reference gradients. This does not alter the active frozen TUNE thread policy or demonstrate whole-audit speedup.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
