"""Bounded numerical fixtures, not generated research examples or model fits.

Compare the native primitive-attention route with CPU FP32/BF16/FP64 for four
operations. Non-unit normalization and projection weights come from the same
preserved trained checkpoint. Inputs/upstream derivatives are seeded fixtures.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA'] = '0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES'] = os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK', '0'))]

import torch
from torch import nn
from torch.nn import functional as F

from scripts.qualify_trained_gradient import compare_gradients


class Operation(nn.Module):
    def __init__(self, kind, state):
        super().__init__()
        self.kind = kind
        names = {'rms': ['attention_norm.weight'],
                 'linear': [],
                 'swiglu': ['feed_forward.w1.weight', 'feed_forward.w2.weight', 'feed_forward.w3.weight'],
                 'sdpa': []}[kind]
        self.weights = nn.ParameterList([nn.Parameter(state['decoder.layers.0.' + name].clone()) for name in names])
        # FusedQKVLinear exports three projection tensors in state_dict while
        # named_parameters exposes one fused tensor. Reconstruct its Q,K,V order.
        if kind == 'linear':
            self.weights.append(nn.Parameter(torch.cat([
                state['decoder.layers.0.attention.qkv_linear.' + name + '.weight']
                for name in ['wq', 'wk', 'wv']], dim=0).clone()))

    def forward(self, *inputs):
        x = inputs[0]
        if self.kind == 'rms':
            return F.rms_norm(x, (768,), self.weights[0], 1e-5)
        if self.kind == 'linear':
            return F.linear(x, self.weights[0])
        if self.kind == 'swiglu':
            return F.linear(F.silu(F.linear(x, self.weights[0])) * F.linear(x, self.weights[2]), self.weights[1])
        return F.scaled_dot_product_attention(*inputs, is_causal=True, enable_gqa=True)


class Objective(nn.Module):
    def __init__(self, operation, device, bf16):
        super().__init__()
        self.operation, self.device, self.bf16 = operation, device, bf16

    def forward(self, upstream, *inputs):
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.bf16):
            output = self.operation(*inputs)
            return (output.float() * upstream).sum(), output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus')  # Standard launcher argument; no corpus is opened.
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True)
    started = time.time()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('Exactly eight independent operator cases required')
    kind = ['rms', 'linear', 'swiglu', 'sdpa'][rank // 2]
    precision = ['fp32', 'bf16'][rank % 2]
    torch.set_num_threads(2)
    torch.manual_seed(41729)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['update'] != 7680 or saved['signature']['depth_policy'] != 'fixed':
        raise ValueError('Unexpected trained fixture state')
    operation = Operation(kind, saved['model'])
    del saved
    inputs = ([torch.randn(1, 12, 1024, 64) * 2, torch.randn(1, 4, 1024, 64) * 2,
               torch.randn(1, 4, 1024, 64)] if kind == 'sdpa' else [torch.randn(1, 1024, 768)])
    with torch.no_grad():
        shape = operation(*inputs).shape
    upstream = torch.randn(shape) / (torch.tensor(shape).prod().item() ** .5)

    def gradients(module, current):
        return {**{name: value.grad.detach().cpu().float().clone() for name, value in module.named_parameters()},
                **{'input_' + str(i): value.grad.detach().cpu().float().clone() for i, value in enumerate(current)}}

    references = {}
    for label in ['fp32', 'bf16', 'fp64']:
        dtype = torch.float64 if label == 'fp64' else torch.float32
        ref = copy.deepcopy(operation).to(dtype=dtype)
        current = [x.to(dtype=dtype).clone().requires_grad_() for x in inputs]
        loss, output = Objective(ref, 'cpu', label == 'bf16')(upstream, *current)
        loss.backward()
        references[label] = {'loss': float(loss.detach()), 'output': output.detach().float(),
                             'output_dtype': str(output.dtype), 'gradients': gradients(ref, current)}
        del ref, current, loss, output
    import torch_neuronx
    torch.neuron.set_device(0)
    operation.to('neuron')
    current = [x.to('neuron').requires_grad_() for x in inputs]
    objective = Objective(operation, 'neuron', precision == 'bf16')
    compiled = torch.compile(objective, backend='neuron', fullgraph=True, dynamic=False)
    loss, output = compiled(upstream.to('neuron'), *current)
    loss.backward()
    actual = gradients(operation, current)
    actual_output = output.detach().cpu().float()
    metrics = {}
    for label, reference in references.items():
        metrics[label] = {'gradient': compare_gradients(actual, reference['gradients']),
                          'output_relative_l2': float((actual_output-reference['output']).norm()/reference['output'].norm().clamp_min(1e-30)),
                          'loss_difference': float(loss.detach().cpu())-reference['loss'],
                          'reference_output_dtype': reference['output_dtype']}
    report = {'status': 'COMPLETE_NUMERICAL_OPERATOR_FIXTURE', 'kind': kind, 'precision': precision,
              'native_output_dtype': str(output.dtype), 'metrics': metrics,
              'native_fallback_ops': sorted(map(str, torch_neuronx.get_fallback_ops())),
              'wall_seconds': time.time()-started, 'optimizer_updates': 0,
              'scope': 'Seeded numerical fixtures, not research data. Primitive SDPA route in every case. Trained layer0 weights are private. No model evaluation or optimizer updates.'}
    (args.output / f'case-{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    if rank == 0:
        while not all((args.output / f'case-{i}.json').exists() for i in range(8)):
            if time.time() > args.deadline_unix:
                raise TimeoutError('Operator fixture deadline')
            time.sleep(1)
        (args.output / 'report.json').write_text(json.dumps({'status': 'COMPLETE_EIGHT_OPERATOR_FIXTURES',
            'cases': [json.loads((args.output / f'case-{i}.json').read_text()) for i in range(8)],
            'wall_seconds': time.time()-started, 'scope': 'Numerical qualification only, no synthetic research corpus.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
