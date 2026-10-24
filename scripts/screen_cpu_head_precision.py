"""Bounded TRAIN-only CPU screen of a prospective precision optimization.

No optimizer updates, held-out scoring, native qualification or active source
changes. A passing subset is only a reason to try full global qualification.
"""
import argparse
import contextlib
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
from compute_trainium.training import PackedCorpus, file_sha256
from scripts.capture_gradient_precision_control import compare_gradients


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
    if file_sha256(a.checkpoint) != a.expected_sha256:
        raise AssertionError('Registered checkpoint changed')
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    sig = saved['signature']
    if saved['update'] != 7680 or sig['depth_policy'] != 'fixed':
        raise ValueError('Only the registered fixed two-billion parent is permitted')
    corpus = a.root/'natural-append-6500m-v2/source.json'
    if file_sha256(corpus) != sig['corpus_sha256']:
        raise ValueError('Registered source changed')
    packed = PackedCorpus.from_mapped(corpus.parent/'packed', 1024,
                                    sig['data_seed'], sig['corpus_sha256'])
    if packed.fingerprint != sig['packing_sha256']:
        raise AssertionError('Packing changed')
    config = ModelConfig(**{k: v for k, v in sig['model'].items()
                            if k in ModelConfig.__dataclass_fields__})
    model = IndependentModel(config, repetitions=2)
    model.load_titan_state(saved['model'])
    # Eight ranks' first routed windows, selected before any new precision scores.
    # This is not the full256-window gradient gate.
    windows = [routed_indices(saved['cursor'], 0, rank, 32, 4, 2,
                              sig['routing_seed'])[0] for rank in range(8)]
    batches = [packed.batch([index], 0)[0] for index in windows]
    target_count = sum(int(b.target_mask.sum()) for b in batches)
    del saved
    reference = None
    rows = []
    for mode in ['fp32', 'bf16_all_matmuls', 'bf16_body_fp32_head']:
        model.zero_grad(set_to_none=True)
        total = 0.
        for batch in batches:
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=mode != 'fp32'):
                hidden = model.hidden(batch.inputs)
                if mode != 'bf16_body_fp32_head':
                    logits = F.linear(hidden, model.embedding.weight)
            if mode == 'bf16_body_fp32_head':
                with torch.autocast('cpu', enabled=False):
                    logits = F.linear(hidden.float(), model.embedding.weight)
            losses = F.cross_entropy(logits.float().reshape(-1, config.vocab_size),
                                     batch.targets.reshape(-1), reduction='none')
            loss = (losses * batch.target_mask.reshape(-1)).sum() / target_count
            loss.backward()
            total += float(loss.detach())
            if time.time() > a.deadline_unix:
                raise TimeoutError('Registered CPU precision-screen budget exhausted')
        gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        if mode == 'fp32':
            reference = gradients
        comparison = compare_gradients(gradients, reference)
        row = {'mode': mode, 'nll': total, 'gradient_vs_cpu_fp32': comparison,
               'elapsed_seconds': time.time()-started}
        rows.append(row)
        write_state(a.output/'progress.json', {'rows': rows})
        print(json.dumps({'mode': mode, 'relative_l2': comparison['relative_l2'],
                          'elapsed_seconds': row['elapsed_seconds']}), flush=True)
        if mode != 'fp32':
            del gradients
    if file_sha256(a.checkpoint) != a.expected_sha256:
        raise AssertionError('Read-only screen changed checkpoint')
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write_state(a.output/'report.json', {
        'status': 'COMPLETE_TRAIN_SUBSET_CPU_PRECISION_SCREEN',
        'rows': rows, 'checkpoint_sha256': a.expected_sha256,
        'source_update': 7680, 'train_windows': len(windows),
        'real_train_targets': target_count, 'native_chips_assigned': 0,
        'optimizer_updates': 0, 'test_accessed': False,
        'wall_seconds': time.time()-started,
        'cpu_seconds': {'user': usage.ru_utime, 'system': usage.ru_stime},
        'scope': 'Independent CPU reference, eight preselected actual TRAIN windows. '
                 'No native fidelity, global qualification, throughput or prediction '
                 'benefit is established. Frozen active FP32 fits are unchanged.'})


if __name__ == '__main__':
    main()
