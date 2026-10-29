"""CPU-only source-location experiment through Torch export and StableHLO.

No model compilation, accelerator, SDK mutation, or research data. This is a
software fixture, not attribution of any scientific run's cache misses.
"""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sysconfig
import time


def main():
    import torch
    from torch_mlir import fx, ir

    start = time.time()
    torch.set_num_threads(1)
    source = Path(sysconfig.get_path('purelib')) / 'torch_neuronx/neuron_dynamo_backend/compile.py'
    raw = source.read_bytes()
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)
             and n.name in ('_serialize_mlir_module', '_content_key')]
    if len(nodes) != 2:
        raise AssertionError('Expected unchanged installed key functions')
    ns = dict(io=io, hashlib=hashlib, os=os,
              _NON_DETERMINISTIC_ATTRS=frozenset({'torch.debug_dump_path'}))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    definition = '''class Fixture(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1.0)
'''
    x = torch.arange(-4, 4, dtype=torch.float32).reshape(2, 4)
    programs, modules, outputs = [], [], []
    for label in ('launch-a', 'launch-b'):
        scope = dict(torch=torch)
        exec(compile(definition, f'/fixture/{label}/model.py', 'exec'), scope)
        model = scope['Fixture']()
        outputs.append(model(x))
        program = torch.export.export(model, (x,), strict=False)
        programs.append(program)
        modules.append(fx.export_and_import(program, output_type='stablehlo'))
    text = [m.operation.get_asm(enable_debug_info=False) for m in modules]
    debug = [m.operation.get_asm(enable_debug_info=True) for m in modules]
    serialized = [ns['_serialize_mlir_module'](m) for m in modules]
    stripped = []
    for m, assembly in zip(modules, text):
        with m.context:
            stripped.append(ns['_serialize_mlir_module'](ir.Module.parse(assembly)))
    result = dict(
        same_eager_output=torch.equal(*outputs),
        same_export_graph=str(programs[0].graph_module.graph) == str(programs[1].graph_module.graph),
        source_locations_survive=[f'/fixture/{label}/model.py' in t
                                  for label, t in zip(('launch-a', 'launch-b'), debug)],
        same_stablehlo_without_locations=text[0] == text[1],
        same_serialized_bytes=serialized[0] == serialized[1],
        same_content_key=ns['_content_key'](serialized[0]) == ns['_content_key'](serialized[1]),
        same_after_reparse_without_locations=stripped[0] == stripped[1],
        sdk_source_sha256=hashlib.sha256(raw).hexdigest(),
        assigned_chips=0, optimizer_updates=0, research_examples=0,
        wall_seconds=time.time() - start,
        scope='CPU export/StableHLO fixture. Not the complete Native Dynamo pipeline, an actual-run miss attribution, or a qualified cache modification.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
