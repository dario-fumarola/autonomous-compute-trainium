"""Explicit-depth independent reference for conventional-control qualification."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TORCH_XLA','0')


def make_reference(config,recurrent):
    if recurrent or config.layers!=8:raise ValueError('Exact conventional eight-layer reference required')
    from scripts.check_ordinary8_cpu import ordinary_reference_config
    from scripts.qualify_parcae_candidate_cpu import PlainTiedHead
    import parcae_lm.models.gpt.gpt as gpt
    gpt.LinearCrossEntropyLoss=PlainTiedHead
    return ordinary_reference_config(config).construct_model().train()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--mode',choices=['audit','replay'],required=True)
    p.add_argument('--warm',type=Path,required=True);p.add_argument('--replay',type=Path);p.add_argument('--gold',type=Path)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time()
    import torch
    from compute_trainium.training import file_sha256
    from compute_trainium.queue import write_state
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen conventional qualification changed')
    plan=json.loads(a.plan.read_text())
    if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Exact independent/reference source changed')
    prep=a.root/plan['preparation_directory']
    if file_sha256(prep/'plan.json')!=plan['preparation_plan_sha256']:raise AssertionError('Prepared ordinary8 plan changed')
    warm=json.loads((a.warm/'report.json').read_text())
    if warm['managed_qualification_plan_sha256']!=a.plan_sha or warm['arm']!='ordinary':raise ValueError('Exact ordinary8 warm result required')
    if a.mode=='audit':
        from scripts import audit_parcae_precise_training_point as engine
        from compute_trainium.parcae_runtime_scalars import changing_qualification_schedule
        engine.make_reference=make_reference;engine.configure_optimizer=changing_qualification_schedule
        result=engine.audit(prep/'audit-root',a.warm,a.upstream,a.output)
        if result['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT':raise RuntimeError('Original first-update/full-global gradient gate failed')
    else:
        if a.replay is None or a.gold is None:raise ValueError('Replay and independent gold required')
        torch.set_num_threads(12);torch.set_num_interop_threads(1)
        from scripts.run_parcae_resumed_training_qualification import check_native_pair
        from compute_trainium.parcae_training_candidate import gradient_comparison
        gold=json.loads((a.gold/'report.json').read_text());replayed=json.loads((a.replay/'report.json').read_text())
        if (gold['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT'
                or replayed['managed_qualification_plan_sha256']!=a.plan_sha):raise ValueError('Pinned successful gold and owned replay required')
        for path,sha in [(a.gold/'cpu-global.pt',gold['cpu_global_gradient_sha256']),
                         (a.replay/'updated-global-native.pt',replayed['updated_global_gradient_sha256'])]:
            if file_sha256(path)!=sha:raise AssertionError('Independent/replay vector changed')
        pair=check_native_pair(a.warm,a.replay)
        if not pair['passes_recovery_gate'] or pair['parameters']!=141733120:raise AssertionError('Original named-state replay gate failed')
        expected=torch.load(a.gold/'cpu-global.pt',map_location='cpu',mmap=True,weights_only=False)
        actual=torch.load(a.replay/'updated-global-native.pt',map_location='cpu',mmap=True,weights_only=False)
        if (expected['source_cursor'],expected['real_targets'])!=(actual['source_cursor'],actual['real_targets']):raise AssertionError('Replay source denominator differs')
        delta=gradient_comparison(expected['gradients'],actual['gradients']);difference=actual['loss']-expected['loss']
        if not delta['passes_two_percent_gate'] or abs(difference)>.003:raise AssertionError('Full replay gradient fails independent FP32 gate')
        a.output.mkdir(parents=True,exist_ok=False)
        write_state(a.output/'report.json',dict(status='PASS_ORDINARY8_NAMED_REPLAY_AND_GLOBAL_GRADIENT',
            plan_sha256=a.plan_sha,pair=pair,gradient=delta,replay_minus_cpu_nll=difference,
            serialized_future_checkpoint_bytes_equal=replayed['final_checkpoint_sha256']==warm['final_checkpoint_sha256'],
            assigned_chips=0,optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,
            scope='Exact restored full state/RNG and original2% named replay update/moment/global-gradient gates. Future trajectory identity remains unestablished.'))


if __name__=='__main__':main()
