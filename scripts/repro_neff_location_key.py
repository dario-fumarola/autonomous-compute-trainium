"""CPU-only fixture for location sensitivity in the installed NEFF key function.

Run in the Native Python environment, without accelerator devices. This does
not compile a model or demonstrate why a particular scientific run missed its
cache. It executes two unchanged functions extracted from the installed source.
"""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sysconfig
import time


def probe(source):
    from torch_mlir import ir

    if os.environ.get('TORCH_NEURONX_NEFF_KEY_FROM_HLO', '1') == '0':
        raise ValueError('This fixture covers the default content-only key path')
    start = time.time()
    data = source.read_bytes()
    tree = ast.parse(data)
    names = {'_serialize_mlir_module', '_content_key'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        raise AssertionError('Expected installed serialization/key implementation')
    namespace = dict(io=io, hashlib=hashlib, os=os,
                     _NON_DETERMINISTIC_ATTRS=frozenset({'torch.debug_dump_path'}))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    with ir.Context():
        modules = [ir.Module.parse(f'module {{}} loc("{name}.py":1:1)')
                   for name in ('launch-a', 'launch-b')]
        raw = [namespace['_serialize_mlir_module'](m) for m in modules]
        text = [m.operation.get_asm(enable_debug_info=False) for m in modules]
        stripped = [namespace['_serialize_mlir_module'](ir.Module.parse(s)) for s in text]
        result = dict(sdk_source_sha256=hashlib.sha256(data).hexdigest(),
            same_without_locations=text[0] == text[1],
            same_serialized_bytes=raw[0] == raw[1],
            same_content_key=namespace['_content_key'](raw[0]) == namespace['_content_key'](raw[1]),
            same_after_reparse_without_locations=stripped[0] == stripped[1],
            assigned_chips=0, optimizer_updates=0, research_examples=0,
            wall_seconds=time.time()-start,
            scope='Numerical software fixture only. Location sensitivity of the current serialization/key functions; not an actual-run cache-miss attribution or a qualified compiler change.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(sysconfig.get_path('purelib')) /
                        'torch_neuronx/neuron_dynamo_backend/compile.py')
    args = parser.parse_args()
    print(json.dumps(probe(args.source), indent=2))
