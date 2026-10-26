"""Independent first update, full gradient and named replay for each baseline."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TORCH_XLA','0')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--profile',choices=['one','mixing','ordinary8'],required=True);p.add_argument('--mode',choices=['audit','replay'],required=True)
    p.add_argument('--warm',type=Path,required=True);p.add_argument('--replay',type=Path);p.add_argument('--gold',type=Path)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time()
    import torch
    from compute_trainium.training import file_sha256
    from compute_trainium.queue import write_state
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Numerical baseline plan changed')
    plan=json.loads(a.plan.read_text());profile=plan['profiles'][a.profile]
    if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Exact numerical/reference source changed')
    prep=a.root/profile['preparation_directory'];reference=a.root/profile['reference_preparation_directory']
    if (file_sha256(prep/'plan.json')!=profile['preparation_plan_sha256']
            or file_sha256(reference/'plan.json')!=profile['reference_plan_sha256']):raise AssertionError('Exact prepared/reference model changed')
    warm=json.loads((a.warm/'report.json').read_text())
    if (warm['managed_qualification_plan_sha256']!=a.plan_sha or warm['qualification_profile']!=a.profile
            or warm['initial_model_digest']!=profile['initial_model_digest']):raise ValueError('Exact baseline warm state required')
    if a.mode=='audit':
        from scripts import audit_parcae_precise_training_point as engine
        from compute_trainium.parcae_runtime_scalars import changing_qualification_schedule
        if a.profile=='ordinary8':
            from scripts.audit_ordinary8_native import make_reference
            engine.make_reference=make_reference
        elif a.profile=='mixing':
            from compute_trainium.lean_mixing_reference import make_reference
            engine.make_reference=make_reference
        engine.configure_optimizer=changing_qualification_schedule
        root=a.root/profile['reference_root_view']
        if (root/'parcae-low-depth-qualification-v1').resolve()!=reference.resolve():raise AssertionError('Independent auditor view changed')
        result=engine.audit(root,a.warm,a.upstream,a.output)
        if result['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT':raise RuntimeError('Original first-update/full-global gradient gate failed')
    else:
        if a.replay is None or a.gold is None:raise ValueError('Owned replay and successful independent gold required')
        torch.set_num_threads(12);torch.set_num_interop_threads(1)
        from scripts.run_parcae_resumed_training_qualification import check_native_pair
        from compute_trainium.parcae_training_candidate import gradient_comparison
        gold=json.loads((a.gold/'report.json').read_text());replayed=json.loads((a.replay/'report.json').read_text())
        if (gold['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT'
                or replayed['managed_qualification_plan_sha256']!=a.plan_sha or replayed['qualification_profile']!=a.profile):
            raise ValueError('Exact completed independent audit and replay required')
        for path,sha in [(a.gold/'cpu-global.pt',gold['cpu_global_gradient_sha256']),
                         (a.replay/'updated-global-native.pt',replayed['updated_global_gradient_sha256'])]:
            if file_sha256(path)!=sha:raise AssertionError('Independent/replayed gradient changed')
        pair=check_native_pair(a.warm,a.replay)
        if not pair['passes_recovery_gate'] or pair['parameters']!=profile['parameters']:raise AssertionError('Original complete named replay gate failed')
        expected=torch.load(a.gold/'cpu-global.pt',map_location='cpu',mmap=True,weights_only=False)
        actual=torch.load(a.replay/'updated-global-native.pt',map_location='cpu',mmap=True,weights_only=False)
        if (expected['source_cursor'],expected['real_targets'])!=(actual['source_cursor'],actual['real_targets']):raise AssertionError('Replay source normalization differs')
        delta=gradient_comparison(expected['gradients'],actual['gradients']);difference=actual['loss']-expected['loss']
        if not delta['passes_two_percent_gate'] or abs(difference)>.003:raise AssertionError('Replay gradient fails independent FP32 gate')
        a.output.mkdir(parents=True,exist_ok=False)
        write_state(a.output/'report.json',dict(status='PASS_BASELINE_NAMED_REPLAY_AND_GLOBAL_GRADIENT',
            plan_sha256=a.plan_sha,profile=a.profile,pair=pair,gradient=delta,replay_minus_cpu_nll=difference,
            serialized_future_checkpoint_bytes_equal=replayed['final_checkpoint_sha256']==warm['final_checkpoint_sha256'],
            assigned_chips=0,optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,
            scope='Exact full state/RNG restoration and original2% named replay update/moment/global-gradient gates. Future trajectory identity remains unestablished.'))


if __name__=='__main__':main()
