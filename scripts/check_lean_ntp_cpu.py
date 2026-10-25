"""Natural-TRAIN CPU parity after removing the unused future head; no launch."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('USE_TORCH_XLA', '0')
import torch

from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_evaluation import prediction_state
from compute_trainium.future_span_math_sdpa import FutureSpanMathSDPACandidate
from compute_trainium.future_span_training import FutureSpanObjective, objective_coefficients, training_window
from compute_trainium.metrics import real_token_losses
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer
from compute_trainium.parcae_work_accounting import global_update_work
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus, file_sha256
from scripts.qualify_native import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    plan = json.loads(args.plan.read_text())
    if plan['identity'] != 'fixed-two-ntp-main-v1' or plan['test_accessed']:
        raise ValueError('Only the current frozen development initializer is eligible')
    if any(file_sha256(path) != sha for path, sha in plan['source_sha256'].items()):
        raise AssertionError('Original scientific sources changed')
    config = ParcaeCandidateConfig(**plan['model'])
    span = FutureSpanConfig(**plan['future_config'])
    initial = plan['initial_states'][plan['cases']['one-2b']['initial_state']]
    source = args.plan.parent / initial['file']
    if file_sha256(source) != initial['sha256']:
        raise AssertionError('Original unfitted initializer changed')
    saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    if digest(dict(sorted(saved['model'].items()))) != initial['model_digest']:
        raise AssertionError('Original unfitted model digest differs')
    # This is a read-only numerical check. Its update must never enter science.
    body = prediction_state(saved['model'])
    packed = PackedCorpus.from_mapped(args.root / plan['data']['packed'], 1024, 23,
                                     plan['data']['corpus_sha256'])
    if packed.fingerprint != plan['data']['packing_fingerprint']:
        raise AssertionError('Original natural source packing changed')
    rows = []
    for name, depth in [('one-2b', 1), ('fixed-2b', 2)]:
        if time.time()-started > 900:
            raise RuntimeError('Finite CPU parity cap')
        available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:'))
        if available < 544*1024**3 or min(shutil.disk_usage('/').free, shutil.disk_usage(args.output).free) < 35*1024**3:
            raise RuntimeError('CPU check resource reserve')
        case = plan['cases'][name]
        if case['objective_weight'] != 0:
            raise ValueError('A positive auxiliary weight cannot be removed')
        full = FutureSpanMathSDPACandidate(config, span, initialize=False, fp16_body=False)
        lean = ParcaePrimitiveSDPACandidate(config, initialize=False, fp16_body=False)
        full.load_state_dict(saved['model'], strict=True)
        lean.load_state_dict(body, strict=True)
        optimizer_full = ParcaeRuntimeScalarOptimizer(full, gate_policy='release_muon', lr=.001, muon_lr=.001)
        optimizer_lean = ParcaeRuntimeScalarOptimizer(lean, gate_policy='release_muon', lr=.001, muon_lr=.001)
        windows = [training_window(packed, i, config, span, case['state_seed']) for i in range(4)]
        targets = sum(int(w[2].sum()) for w in windows)
        anchors = sum(int(w[-1].sum()) for w in windows)
        coefficients = objective_coefficients(targets, anchors, 0., 1)
        objective = FutureSpanObjective(full)
        losses = [0., 0.]
        for values in windows:
            a, _ = objective(*values, coefficients, depth, depth)
            logits = lean(values[0], values[3], total_steps=depth, backward_cap=depth)
            b = real_token_losses(logits, values[1], values[2]).sum()*coefficients[0]
            if not torch.equal(a.detach(), b.detach()):
                raise AssertionError('CPU ordered NTP objective differs')
            losses[0] += a.item(); losses[1] += b.item()
            a.backward(); b.backward()
            del a, b, logits
        full_named, lean_named = dict(full.named_parameters()), dict(lean.named_parameters())
        if any(not torch.equal(full_named[k].grad, v.grad) for k, v in lean_named.items()):
            raise AssertionError('CPU prediction gradients differ')
        if any(v.grad is None or torch.count_nonzero(v.grad) for k, v in full_named.items()
               if k.startswith('future_lm_head.')):
            raise AssertionError('Unused auxiliary head has a nonzero or absent gradient')
        norm_full = torch.nn.utils.clip_grad_norm_(full.parameters(), 1.)
        norm_lean = torch.nn.utils.clip_grad_norm_(lean.parameters(), 1.)
        # A different list length can change the norm reduction's last bits.
        if not torch.isclose(norm_full, norm_lean, rtol=1e-6, atol=1e-7):
            raise AssertionError('Removing zero gradients changed clipping materially')
        optimizer_full.step(); optimizer_lean.step()
        maximum = max(float((full_named[k]-v).abs().max()) for k, v in lean_named.items())
        for k, v in lean_named.items():
            torch.testing.assert_close(full_named[k], v, rtol=1e-6, atol=1e-7)
        work = global_update_work(lean, 1024, 32, 1, 16, gate_policy='release_muon',
                                 forward_steps=depth, backward_cap=depth)
        rows.append(dict(case=name, depth=depth, natural_training_windows=4, real_targets=targets,
                         full_loss=losses[0], lean_loss=losses[1], prediction_gradients_bit_equal=True,
                         zero_auxiliary_gradients=True, full_gradient_norm=float(norm_full),
                         lean_gradient_norm=float(norm_lean), maximum_updated_parameter_absolute_error=maximum,
                         deployed_parameters=sum(v.numel() for v in lean.parameters()),
                         training_parameters_removed=sum(v.numel() for k,v in full_named.items() if k.startswith('future_lm_head.')),
                         lean_global_update_work=work, qualification_only=True))
        del windows, objective, optimizer_full, optimizer_lean, full_named, lean_named, full, lean
        gc.collect()
    if file_sha256(source) != initial['sha256']:
        raise AssertionError('Numerical check modified the scientific initializer')
    write_state(args.output/'report.json', dict(status='PASS_LEAN_NTP_CPU_PARITY',
        plan_sha256=file_sha256(args.plan), initializer_sha256=initial['sha256'],
        source_sha256=file_sha256(__file__), records=rows, assigned_chips=0,
        actual_qualification_optimizer_updates=4, scientific_updates=0,
        test_accessed=False, wall_seconds=time.time()-started,
        scope='Four original TRAIN windows per arm,CPU FP32 objective/gradient and one optimizer update per full/lean model. No native,FP16,full512-window,optimizer-state-recovery,throughput or trajectory qualification. No scientific state is saved or adopted;no new quality result.'))


if __name__ == '__main__':
    main()
