"""CPU-only large factory agreement on explicitly non-research fixtures."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from compute_trainium.model import ModelConfig
from compute_trainium.reinjection_control import (ReinjectionControlConfig,
    build_reinjection_control, IndependentReinjectionControl)
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.capture_gradient_precision_control import compare_gradients


def grouped_qkv_gradient(q, k, v, kv_heads, head_dim):
    """TorchTitan packs Q/K/V inside each KV group, not in three flat spans."""
    dim = q.shape[1]
    per_group = q.shape[0]//(kv_heads*head_dim)
    return torch.cat([q.reshape(kv_heads, per_group, head_dim, dim),
                      k.reshape(kv_heads, 1, head_dim, dim),
                      v.reshape(kv_heads, 1, head_dim, dim)], dim=1).reshape(-1, dim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    started = time.time()
    torch.manual_seed(791)
    c = ReinjectionControlConfig()
    native = build_reinjection_control(c)
    base = ModelConfig(**{k: getattr(c, k) for k in ModelConfig.__dataclass_fields__})
    reference = IndependentReinjectionControl(base)
    reference.load_titan_state(native.state_dict())
    tokens = torch.arange(31)[None]
    target = torch.arange(1, 32)[None]
    left, right = native(tokens), reference(tokens)
    loss_left = F.cross_entropy(left.flatten(0, 1), target.flatten())
    loss_right = F.cross_entropy(right.flatten(0, 1), target.flatten())
    loss_left.backward()
    loss_right.backward()
    expected = {'decoder.tok_embeddings.weight': reference.embedding.weight.grad,
                'decoder.norm.weight': reference.norm.weight.grad}
    legacy = dict(expected)
    for i, block in enumerate(reference.blocks):
        prefix = f'decoder.layers.{i}.'
        key = prefix+'attention.qkv_linear.wqkv.weight'
        expected[key] = grouped_qkv_gradient(block.q.weight.grad, block.k.weight.grad,
                                             block.v.weight.grad, c.kv_heads, c.dim//c.heads)
        legacy[key] = torch.cat([block.q.weight.grad, block.k.weight.grad, block.v.weight.grad])
        for suffix, value in [('attention.wo.weight', block.o.weight),
            ('feed_forward.w1.weight', block.gate.weight),
            ('feed_forward.w2.weight', block.down.weight),
            ('feed_forward.w3.weight', block.up.weight),
            ('attention_norm.weight', block.attn_norm.weight),
            ('ffn_norm.weight', block.ffn_norm.weight)]:
            expected[prefix+suffix] = value.grad
            legacy[prefix+suffix] = value.grad
    actual = {n: v.grad for n, v in native.named_parameters()}
    comparison = compare_gradients(actual, expected)
    delta = abs(float(loss_left.detach())-float(loss_right.detach()))
    passes = comparison['relative_l2'] <= 1e-4 and delta <= 1e-5
    report = {
        'status': ('PASS' if passes else 'FAIL')+'_INDEPENDENT_CPU_REINJECTION_FACTORY_FIXTURE',
        'unique_parameters': c.parameter_count(),
        'fixture_positions': 31, 'gradient_comparison': comparison,
        'legacy_incorrect_flat_concat_comparison': compare_gradients(actual, legacy),
        'loss_difference': delta, 'native_chips_assigned': 0,
        'wall_seconds': time.time()-started,
        'source_sha256': {f: file_sha256(f) for f in ['scripts/verify_reinjection_factory.py',
            'compute_trainium/reinjection_control.py', 'compute_trainium/independent.py']},
        'scope': 'Software arange arrays, not research examples. Full150M CPU factory '
                 'versus independent projections/RoPE/blocks. No training, '
                 'held-out performance, native execution or throughput claim.'}
    write_state(a.output/'report.json', report)
    print(json.dumps(report), flush=True)
    if not passes:
        raise AssertionError('CPU native-factory/independent fixture mismatch')


if __name__ == '__main__':
    main()
