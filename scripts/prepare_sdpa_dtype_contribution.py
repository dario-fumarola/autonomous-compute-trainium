"""Prepare an UNAPPLIED source-gating proposal and test CPU metadata only."""
import argparse
import ast
import difflib
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch


def gates(source):
    names = {'_nki_flash_attention_shapes_supported', '_can_use_nki_flash_attention',
             '_can_use_nki_flash_attention_backward'}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names: raise ValueError('Expected installed gate definitions missing')
    namespace = {'torch': torch, 'math': math, '_getenv_bool': lambda name, default: True,
                 'get_logical_neuron_cores': lambda: 2, '_NKI_ATTENTION_SUPPORTS_V_HEAD_DIM': True}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<isolated-cpu-gates>', 'exec'), namespace)
    return namespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--staging', type=Path, required=True)
    a = p.parse_args()
    original = a.source.read_text()
    before = hashlib.sha256(a.source.read_bytes()).hexdigest()
    marker = 'and query.dtype in (torch.bfloat16, torch.float32)'
    if original.count(marker) != 1: raise ValueError('Pinned shared dtype predicate differs')
    proposed = original.replace(marker, 'and query.dtype == torch.bfloat16', 1)
    proposed = proposed.replace(
        '# The forward kernel converts FP16 input tiles to BF16. Use the math path for FP16.',
        '# NKI uses BF16 for non-softmax math. Keep FP32/FP16 on the math path;\n'
        '    # callers can explicitly opt into BF16 through autocast.', 1)
    a.output.mkdir(parents=True, exist_ok=True)
    a.staging.mkdir(exist_ok=False)
    relative = 'torch_neuronx/neuron_dynamo_backend/decompositions.py'
    staged = a.staging/relative
    staged.parent.mkdir(parents=True)
    staged.write_text(original)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), proposed.splitlines(True),
                    fromfile='a/'+relative, tofile='b/'+relative))
    patch_path = a.output/'bf16-only-default-sdpa.patch'
    patch_path.write_text(patch)
    subprocess.run(['git', 'apply', '--check', str(patch_path.resolve())], cwd=a.staging, check=True)
    subprocess.run(['git', 'apply', str(patch_path.resolve())], cwd=a.staging, check=True)
    if staged.read_text() != proposed: raise AssertionError('Patch did not reproduce proposed source')
    old, new = gates(original), gates(proposed)
    def check(namespace, dtype, *, key_dtype=None, scale=None, enabled=True, length=1024):
        namespace['_getenv_bool'] = lambda name, default: enabled
        q = torch.empty((1, 12, length, 64), dtype=dtype, device='meta')
        k = torch.empty((1, 4, length, 64), dtype=key_dtype or dtype, device='meta')
        v = torch.empty_like(k)
        return {'forward': namespace['_can_use_nki_flash_attention'](q, k, v, scale=scale),
                'backward': namespace['_can_use_nki_flash_attention_backward'](q, k, v, scale=scale, is_causal=True)}
    rows = []
    for dtype in [torch.bfloat16, torch.float32, torch.float16]:
        left, right = check(old, dtype), check(new, dtype)
        expected = dtype == torch.bfloat16
        if right != {'forward': expected, 'backward': expected}:
            raise AssertionError('Proposed forward/backward dtype policy differs')
        if dtype == torch.bfloat16 and left != right: raise AssertionError('BF16 path changed')
        rows.append({'case': str(dtype), 'installed_gate': left, 'proposed_gate': right})
    for name, kwargs in [('disabled', {'enabled': False}), ('scale_zero', {'scale': 0.}),
                         ('mixed_dtype', {'key_dtype': torch.float32}), ('length1152', {'length': 1152})]:
        left, right = check(old, torch.bfloat16, **kwargs), check(new, torch.bfloat16, **kwargs)
        if left != right: raise AssertionError('Unrelated gate behavior changed')
        rows.append({'case': name, 'installed_gate': left, 'proposed_gate': right})
    if hashlib.sha256(a.source.read_bytes()).hexdigest() != before:
        raise AssertionError('Installed SDK source changed')
    report = {'status': 'PASS_LOCAL_PATCH_APPLICABILITY_AND_CPU_GATE_FIXTURES',
              'installed_source_sha256': before,
              'proposed_source_sha256': hashlib.sha256(proposed.encode()).hexdigest(),
              'patch_sha256': hashlib.sha256(patch_path.read_bytes()).hexdigest(),
              'cases': rows, 'installed_source_unchanged': True,
              'native_chips_assigned': 0, 'upstream_submission': False,
              'scope': 'Unapplied local policy proposal. CPU meta-tensor eligibility checks and clean application to a private source copy only. No patched native forward/backward, graph-cache or throughput qualification; no change to active experiments.'}
    (a.output/'local-fixture-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'cases': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
