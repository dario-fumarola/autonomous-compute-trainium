"""TRAIN-only grammar inventory, not a verified task or model evaluation."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import resource
import shutil
import time
import warnings

import numpy as np
from tokenizers import Tokenizer

from compute_trainium.authored_code_assertions import literal, Unsupported
from compute_trainium.natural_code_context import node_span, token_span
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID = 'authored-code-grammar-inventory-v1'
PUBLIC = Path('results/authored_code_targets_20261030/train-inventory-v1')


def inventory(source, offsets):
    """Count syntax only. A candidate is NEVER a validated authored answer."""
    if len(source) > 100000:
        raise Unsupported('source_length')
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    if len(nodes) > 12000:
        raise Unsupported('AST_size')
    counts, candidates = Counter(), []
    definitions = {}
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef):
            definitions.setdefault(statement.name, []).append(statement)
    counts['function_definitions'] = sum(isinstance(n, ast.FunctionDef) for n in nodes)
    counts['assertions'] = sum(isinstance(n, ast.Assert) for n in nodes)
    counts['top_level_assertions'] = sum(isinstance(n, ast.Assert) for n in tree.body)
    counts['doctest_prompt_lines'] = sum(line.lstrip().startswith('>>> ') for line in source.splitlines())
    for node in nodes:
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1):
            continue
        counts['equality_assertions'] += 1
        try:
            literal(test.comparators[0])
        except Unsupported:
            continue
        counts['integer_expected_equality'] += 1
        if not isinstance(test.left, ast.Call) or not isinstance(test.left.func, ast.Name):
            continue
        counts['integer_expected_named_call'] += 1
        functions = [f for f in definitions.get(test.left.func.id, []) if f.end_lineno < node.lineno]
        if len(functions) != 1:
            continue
        counts['integer_expected_prior_local_definition'] += 1
        function = functions[0]
        try:
            if test.left.keywords:
                raise Unsupported('keywords')
            for arg in test.left.args:
                literal(arg)
        except Unsupported:
            continue
        counts['literal_arguments_prior_local_definition'] += 1
        fs, fe = node_span(source, function)
        ts, te = node_span(source, test.comparators[0])
        fspan, target = token_span(source, offsets, fs, fe), token_span(source, offsets, ts, te)
        if fspan is None or target is None:
            counts['alignment_exclusion'] += 1
            continue
        if target[1] - fspan[0] > 513:
            counts['definition_outside_512_context'] += 1
            continue
        counts['unverified_context_candidates'] += 1
        kinds = Counter(type(n).__name__ for n in ast.walk(function))
        for kind in ('For', 'While', 'If', 'Call', 'BinOp', 'Subscript', 'Attribute'):
            counts['candidate_body_has_' + kind] += int(kinds[kind] > 0)
        counts['candidate_top_level_assertion'] += int(node in tree.body)
        counts['candidate_definition_annotations'] += int(function.returns is not None or
            any(a.annotation is not None for a in [*function.args.posonlyargs, *function.args.args]))
        candidates.append(dict(target_tokens=list(target), function_tokens=list(fspan),
                               top_level=node in tree.body, body_node_counts=dict(kinds)))
    return counts, candidates


def run(root):
    started = time.time()
    out = root / ID
    if out.exists() or PUBLIC.exists():
        raise FileExistsError('Preserve finite audit attempts')
    data = root / 'indexed-natural-sources-v1'
    accepted = json.loads((data / 'report.json').read_text())
    if accepted['status'] != 'PASS_INDEXED_NATURAL_SOURCES' or accepted['protected_group_overlap']:
        raise AssertionError('Accepted TRAIN-only data required')
    recipe = data / 'PRIVATE-training-documents.jsonl'
    token_path = root / accepted['source_relative'] / 'code.u16'
    tokenizer_path = Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    files = ['scripts/inventory_authored_code_targets.py', 'tests/test_authored_code_inventory.py',
             'compute_trainium/authored_code_assertions.py', 'compute_trainium/natural_code_context.py']
    plan = dict(identity=ID, registered_unix=started, absolute_deadline_unix=started+600,
        maximum_code_documents=10000, maximum_rss_bytes=8*1024**3,
        source_sha256={f:file_sha256(f) for f in files},
        accepted_report_sha256=file_sha256(data/'report.json'),
        recipe_sha256=accepted['files'][recipe.name], code_tokens_sha256=accepted['raw_token_sha256']['code'],
        tokenizer_sha256=file_sha256(tokenizer_path), assigned_chips=0, optimizer_updates=0,
        selection='First10000 accepted TRAIN code documents, unchanged order, exact tokenizer roundtrip. '
            'Count integer equality assertions over literal-argument calls to prior local top-level function definitions, '
            'including nested assertions and control flow. Original definition and answer must fit512-source context. '
            'Syntax counts only: no function interpretation, semantic verification or evaluation set creation.',
        scope='TRAIN parser-coverage diagnostic; unverified candidates are not research examples. No model scores, held-out access or fit admission.')
    out.mkdir(mode=0o700); PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', plan); write_state(PUBLIC/'plan.json', plan)
    counts, rows, groups = Counter(), [], set()
    report = dict(status='STOPPED_AUTHORED_CODE_GRAMMAR_INVENTORY')

    def guard():
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (time.time()>plan['absolute_deadline_unix'] or available<512*1024**3
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>plan['maximum_rss_bytes']
                or min(shutil.disk_usage(root).free, shutil.disk_usage('/').free)<35*1024**3):
            raise RuntimeError('Finite inventory resource guard')

    try:
        guard()
        if file_sha256(recipe)!=plan['recipe_sha256'] or file_sha256(token_path)!=plan['code_tokens_sha256']:
            raise AssertionError('Accepted TRAIN bytes changed')
        tokens=np.memmap(token_path,dtype='<u2',mode='r'); tokenizer=Tokenizer.from_file(str(tokenizer_path))
        with recipe.open() as stream:
            for line in stream:
                meta=json.loads(line)
                if meta['domain']!='code': continue
                if counts['code_documents']>=plan['maximum_code_documents']: break
                guard()
                if int(meta['group'][:8],16)%10==0: raise AssertionError('TRAIN bucket only')
                counts['code_documents']+=1
                lo,n=meta['source_offset'],meta['source_tokens']; ids=tokens[lo:lo+n].tolist()
                text=tokenizer.decode(ids,skip_special_tokens=False); enc=tokenizer.encode(text,add_special_tokens=False)
                if enc.ids!=ids:
                    counts['token_roundtrip_exclusion']+=1; continue
                counts['token_roundtrip_exact']+=1
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',SyntaxWarning)
                        features,candidates=inventory(text,enc.offsets)
                except (SyntaxError,ValueError,RecursionError,MemoryError) as error:
                    counts['bounded_parser_exclusion' if isinstance(error,Unsupported) else 'unparseable_python']+=1
                    continue
                counts['parseable_bounded_python']+=1; counts.update(features)
                if candidates:
                    groups.add(meta['group'])
                    rows.extend(dict(document_id=meta['id'],group=meta['group'],source_offset=lo,source_tokens=n,**c) for c in candidates)
                if counts['code_documents']%1000==0:
                    write_state(out/'progress.json',dict(documents=counts['code_documents'],seconds=time.time()-started))
        guard()
        if any(file_sha256(f)!=h for f,h in plan['source_sha256'].items()): raise AssertionError('Frozen source changed')
        write_state(out/'PRIVATE-unverified-candidates.json',rows)
        report.update(status='COMPLETE_AUTHORED_CODE_GRAMMAR_INVENTORY',counts=dict(counts),candidate_source_groups=len(groups),
            private_candidates_sha256=file_sha256(out/'PRIVATE-unverified-candidates.json'))
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error)); raise
    finally:
        report.update(wall_seconds=time.time()-started,assigned_chips=0,optimizer_updates=0,model_scores=0,heldout_reads=0,
                      plan_sha256=file_sha256(out/'plan.json'),scope=plan['scope'])
        write_state(out/'report.json',report); write_state(PUBLIC/'report.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); run(p.parse_args().root)
