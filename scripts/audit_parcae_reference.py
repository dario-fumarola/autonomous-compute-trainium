"""Audit pinned public reference code; no accelerator or research corpus access."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from compute_trainium.queue import write_state


PIN = '69284c13746e849104f738d6d1a347b1f457df76'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sampling_probe(path, mean, cap, count=131072):
    tree = ast.parse(path.read_text())
    parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Parcae')
    method = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == '_sample_batch_depths')
    method.decorator_list = []
    fixture_class = ast.ClassDef(name='PinnedSampler', bases=[], keywords=[], body=[method], decorator_list=[])
    fixture = ast.fix_missing_locations(ast.Module(body=[fixture_class], type_ignores=[]))
    ns = {'torch': torch, 'math': math, 'Optional': __import__('typing').Optional}
    exec(compile(fixture, str(path), 'exec'), ns)
    sampler = ns['PinnedSampler']()
    sampler.config = SimpleNamespace(sampling_scheme='poisson-truncated-full',
        mean_recurrence=mean, mean_backprop_depth=cap, lockstep_n=False)
    sampler.step = 0
    prefix, tail = sampler._sample_batch_depths(count)
    total = prefix+tail
    if not torch.equal(tail, total.clamp(max=cap)) or int(total.min()) < 1:
        raise AssertionError('Pinned reference sampling changed')
    # The published training configuration uses32 sequences per microbatch.
    # Dense masked execution runs to the maxima, not each sequence's depth.
    executed = prefix.reshape(-1, 32).max(1).values+tail.reshape(-1, 32).max(1).values
    p = math.exp(-mean)
    cdf = p
    expected_tail = 1.
    for j in range(2, cap+1):
        p *= mean/(j-1)
        cdf += p
        expected_tail += 1-cdf
    exact_total = mean+math.exp(-mean)  # zero Poisson draws become one
    if abs(float(total.double().mean())-exact_total) > .04:
        raise AssertionError('Sampled total is inconsistent with clamped Poisson')
    return {'requested_mean': mean, 'backward_cap': cap, 'software_draws': count,
        'actual_expected_forward': exact_total, 'actual_expected_backward': expected_tail,
        'observed_mean_forward': float(total.double().mean()),
        'observed_mean_backward': float(tail.double().mean()),
        'fraction_with_more_than_one_pass': float((total>1).double().mean()),
        'observed_mean_dense_forward_calls_at_microbatch32': float(executed.double().mean()),
        'scope': 'Pinned sampler and masked-loop execution semantics, not measured hardware work or a reconstruction of paper runs.'}


def head_cost_fixture(cost_module):
    """Real tied projection on meta tensors, avoiding any trained model/data."""
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.wte = nn.Embedding(31, 8, device='meta')
            self.transformer.h = nn.ModuleList([nn.Linear(8, 8, bias=False, device='meta')])
            self.lm_head = nn.Linear(8, 31, bias=False, device='meta')
            self.lm_head.weight = self.transformer.wte.weight
    config = SimpleNamespace(block_size=4, n_embd=8, num_attention_heads=2)
    result = cost_module.estimate_flops_gpt(Tiny(), config)
    expected_dense = 6*(8*8+31*8)
    reported_dense = 6*result.total_params
    if reported_dense != 6*8*8 or expected_dense-reported_dense != 6*31*8:
        raise AssertionError('Pinned tied-head fixture differs from inspected estimator')
    return {'unique_parameters': 8*8+31*8,
        'reference_estimated_dense_training_flops_per_position': reported_dense,
        'dense_core_plus_tied_output_projection_training_flops_per_position': expected_dense,
        'omitted_tied_output_projection_matmuls': expected_dense-reported_dense,
        'scope': 'Isolated software fixture proves tied-head omission in the release estimator. It does not quantify net error in the released140M model: value-embedding gathers and other operations require separate accounting.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.set_num_threads(2)
    names = ['LICENSE', 'parcae_lm/modules/injection.py', 'parcae_lm/models/parcae/init.py',
        'parcae_lm/models/parcae/config.py', 'parcae_lm/models/parcae/parcae.py',
        'parcae_lm/configs/parcae/parcae-small-140m.py', 'parcae_lm/configs/gpt/gpt-small-140m.py',
        'launch_configs/parcae-small-140m.yaml', 'runs/sweep_recurrence.sh', 'scripts/cost.py', 'scripts/train.py']
    hashes = {n: hashlib.sha256((a.reference/n).read_bytes()).hexdigest() for n in names}
    provenance = json.loads(Path('third_party/parcae_reference/provenance.json').read_text())
    if provenance['commit'] != PIN:
        raise AssertionError('Pinned upstream identity changed')
    for row in provenance['files']:
        source = (a.reference/row['path']).read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
        if blob != row['git_blob_sha1']:
            raise AssertionError('Reference differs from connector-fetched Git object: '+row['path'])
    cost = load_module(a.reference/'scripts/cost.py', '_pinned_parcae_cost')
    rows = [sampling_probe(a.reference/'parcae_lm/models/parcae/parcae.py', mean, (mean+1)//2)
            for mean in [1, 2, 4, 8]]
    sweep = (a.reference/'runs/sweep_recurrence.sh').read_text()
    config = (a.reference/'launch_configs/parcae-small-140m.yaml').read_text()
    if 'RECURRENCES=(1 2 4 6 8 10 12)' not in sweep or 'model_overwrite.sampling_scheme' in sweep:
        raise AssertionError('Release sweep no longer matches reviewed baseline ambiguity')
    if 'micro_batch_size: 32' not in config:
        raise AssertionError('Inspected microbatch changed')
    result = {'status': 'COMPLETE_PUBLIC_REFERENCE_READINESS_AUDIT', 'upstream_commit': PIN,
        'upstream_repository': 'https://github.com/sandyresearch/parcae', 'source_sha256': hashes,
        'sampling': rows, 'tied_head_fixture': head_cost_fixture(cost),
        'source_transfer_note': 'Pinned sources fetched through the GitHub connector. A first oversized shell argument was rejected before writing train.py; it was transferred in three checked chunks. No package installation or accelerator use.',
        'prior_preparation_failure': 'The first CPU audit completed its calculations but could not save because the output directory was missing. The writer now creates the directory; no training, datasets or held-out scores were involved. That failed attempt has no exact wall receipt.',
        'optimizer_updates': 0, 'assigned_chips': 0, 'test_accessed': False,
        'scope': 'Resolve implementation/accounting details before a faithful future reproduction. Does not invalidate or remeasure published model results and changes no active experiment.',
        'wall_seconds_after_imports': time.time()-started}
    write_state(a.output/'report.json', result)
    print(json.dumps({k:result[k] for k in ['status','sampling','tied_head_fixture','wall_seconds_after_imports']}, indent=2))


if __name__ == '__main__':
    main()
