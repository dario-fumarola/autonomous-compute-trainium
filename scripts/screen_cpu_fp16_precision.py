"""TRAIN-only FP16 feasibility screen; no native execution or model updates.

Compare a fixed eight-window gradient with CPU FP32. A subset pass does not
qualify native operators, a full training batch, scaling recovery or speed.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from compute_trainium.depth_mixture import routed_indices
from compute_trainium.independent import IndependentModel
from compute_trainium.model import ModelConfig
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus, atomic_checkpoint, file_sha256
from scripts.capture_gradient_precision_control import compare_gradients
from scripts.qualify_native import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    started = time.time()
    rows = []
    report = None
    try:
        if file_sha256(a.checkpoint) != a.expected_sha256:
            raise AssertionError('Registered checkpoint changed')
        saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
        sig = saved['signature']
        if saved['update'] != 7680 or sig['depth_policy'] != 'fixed':
            raise ValueError('Only the registered fixed two-billion parent is permitted')
        source = a.root/'natural-append-6500m-v2/source.json'
        if file_sha256(source) != sig['corpus_sha256']:
            raise ValueError('Source changed')
        packed = PackedCorpus.from_mapped(source.parent/'packed', 1024,
                                          sig['data_seed'], sig['corpus_sha256'])
        if packed.fingerprint != sig['packing_sha256']:
            raise AssertionError('Packing changed')
        c = ModelConfig(**{k: v for k, v in sig['model'].items()
                           if k in ModelConfig.__dataclass_fields__})
        model = IndependentModel(c, repetitions=2)
        model.load_titan_state(saved['model'])
        before = digest(model.state_dict())
        indices = [routed_indices(saved['cursor'], 0, rank, 32, 4, 2,
                                  sig['routing_seed'])[0] for rank in range(8)]
        batches = [packed.batch([index], 0)[0] for index in indices]
        count = sum(int(b.target_mask.sum()) for b in batches)
        del saved
        reference = None
        for mode, scale in [('fp32', 1.), ('fp16_fp32_head', 1.),
                            ('fp16_fp32_head_scaled8192', 8192.)]:
            model.zero_grad(set_to_none=True)
            total = 0.
            for batch in batches:
                with torch.autocast('cpu', dtype=torch.float16, enabled=mode != 'fp32'):
                    hidden = model.hidden(batch.inputs)
                # The tied vocabulary projection and CE stay FP32 in every mode.
                with torch.autocast('cpu', enabled=False):
                    logits = F.linear(hidden.float(), model.embedding.weight)
                    loss = (F.cross_entropy(logits.reshape(-1, c.vocab_size),
                        batch.targets.reshape(-1), reduction='none')*
                        batch.target_mask.reshape(-1)).sum()/count
                (loss*scale).backward()
                total += float(loss.detach())
                if time.time() > a.deadline_unix:
                    raise TimeoutError('CPU precision-screen budget exhausted')
            gradients = {name: value.grad.detach().clone()/scale
                         for name, value in model.named_parameters()}
            finite = all(torch.isfinite(g).all().item() for g in gradients.values())
            if mode == 'fp32': reference = gradients
            comparison = compare_gradients(gradients, reference) if finite else None
            row = {'mode': mode, 'loss_scale': scale, 'nll': total,
                   'all_gradients_finite': finite, 'comparison_to_cpu_fp32': comparison,
                   'elapsed_seconds': time.time()-started}
            rows.append(row)
            write_state(a.output/'progress.json', {'rows': rows})
            if mode in ['fp32', 'fp16_fp32_head_scaled8192'] and finite:
                atomic_checkpoint(a.output/(mode+'-private-gradients.pt'),
                                  {'gradients': gradients, 'loss': total})
            print(json.dumps({'mode': mode, 'relative_l2': comparison['relative_l2'] if finite else None,
                              'finite': finite, 'elapsed_seconds': row['elapsed_seconds']}), flush=True)
            if mode != 'fp32': del gradients
        if digest(model.state_dict()) != before or file_sha256(a.checkpoint) != a.expected_sha256:
            raise AssertionError('Read-only screen changed parameters or source')
        report = {'status': 'COMPLETE_TRAIN_SUBSET_CPU_FP16_SCREEN', 'rows': rows,
                  'checkpoint_sha256': a.expected_sha256, 'source_update': 7680,
                  'train_windows': 8, 'real_train_targets': count,
                  'parameters_unchanged': True, 'native_chips_assigned': 0,
                  'optimizer_updates': 0, 'test_accessed': False,
                  'scope': 'Independent CPU reference on eight pinned actual TRAIN windows. '
                           'No native operator, full256-window gradient, optimizer update, '
                           'checkpoint scaling recovery, speed or learning qualification. '
                           'Active FP32 scientific fits stay frozen.'}
    except BaseException as error:
        report = {'status': 'STOPPED_TRAIN_SUBSET_CPU_FP16_SCREEN', 'error': str(error),
                  'rows': rows, 'native_chips_assigned': 0, 'optimizer_updates': 0,
                  'test_accessed': False}
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(wall_seconds=time.time()-started,
                      cpu_seconds={'user': usage.ru_utime, 'system': usage.ru_stime})
        write_state(a.output/'report.json', report)


if __name__ == '__main__':
    main()
