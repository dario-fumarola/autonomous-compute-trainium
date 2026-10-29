"""Aggregate syntax census on the same previously inspected TRAIN prefix.

No source execution, target generation, model evaluation or selector changes.
"""
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

from compute_trainium.authored_code_assertions import Unsupported, literal
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID = 'train-code-assertion-census-v1'
PUBLIC = Path('results/local_arithmetic_assertions_20261029/syntax-census-v1')


def integer(node):
    try:
        literal(node)
        return True
    except Unsupported:
        return False


def census(source):
    if len(source) > 100000:
        raise Unsupported('source_length')
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    if len(nodes) > 12000:
        raise Unsupported('AST_size')
    counts = Counter()
    for node in nodes:
        if isinstance(node, ast.Assert):
            counts['assert_statements'] += 1
            test = node.test
            if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
                counts['assert_single_equality'] += 1
                if integer(test.comparators[0]):
                    counts['assert_equality_integer_rhs'] += 1
                    counts['assert_integer_rhs_with_message'] += int(node.msg is not None)
                    lhs = list(ast.walk(test.left))
                    counts['integer_assert_lhs_has_call'] += int(any(isinstance(n, ast.Call) for n in lhs))
                    counts['integer_assert_lhs_has_name'] += int(any(isinstance(n, ast.Name) for n in lhs))
                    counts['integer_assert_lhs_two_arithmetic_nodes'] += int(sum(isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)) for n in lhs) >= 2)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'assertEqual':
            counts['assertEqual_calls'] += 1
            if len(node.args) >= 2 and (integer(node.args[0]) or integer(node.args[1])):
                counts['assertEqual_integer_operand'] += 1
        if isinstance(node, ast.Constant) and type(node.value) is str and '>>>' in node.value:
            counts['string_constants_containing_doctest_prompt'] += 1
    counts['documents_with_assert'] = int(counts['assert_statements'] > 0)
    counts['documents_with_integer_rhs_assert'] = int(counts['assert_equality_integer_rhs'] > 0)
    counts['documents_with_assertEqual'] = int(counts['assertEqual_calls'] > 0)
    return counts


def run(root):
    start = time.time()
    out = root/ID
    if out.exists() or PUBLIC.exists():
        raise FileExistsError('Preserve completed census')
    previous = root/'local-arithmetic-assertion-train-audit-v1/plan.json'
    old = json.loads(previous.read_text())
    accepted_path = root/'indexed-natural-sources-v1/report.json'
    accepted = json.loads(accepted_path.read_text())
    if accepted['status'] != 'PASS_INDEXED_NATURAL_SOURCES' or accepted['protected_group_overlap']:
        raise AssertionError('Accepted TRAIN-only source required')
    recipe = root/'indexed-natural-sources-v1/PRIVATE-training-documents.jsonl'
    tokens_path = root/accepted['source_relative']/'code.u16'
    tokenizer_path = Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    plan = dict(identity=ID, registered_unix=start, deadline_unix=start+180,
        maximum_code_documents=10000, maximum_rss_bytes=8*1024**3,
        previous_plan_sha256=file_sha256(previous), source_sha256=file_sha256(__file__),
        accepted_report_sha256=old['accepted_report_sha256'], recipe_sha256=old['recipe_sha256'],
        code_tokens_sha256=old['code_tokens_sha256'], tokenizer_sha256=old['tokenizer_sha256'],
        scope='Descriptive syntax census on the same 10000 TRAIN code documents as the completed zero-coverage audit. Count assertion forms, not verified computations. No targets, model scores, source publication, held-out reads, selector revision or fit admission.')
    out.mkdir(mode=0o700)
    PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', plan)
    write_state(PUBLIC/'plan.json', plan)
    counts = Counter()
    result = dict(status='STOPPED_TRAIN_CODE_ASSERTION_CENSUS')

    def guard():
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (time.time() > plan['deadline_unix'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > plan['maximum_rss_bytes']
                or available < 512*1024**3 or min(shutil.disk_usage(root).free, shutil.disk_usage('/').free) < 35*1024**3):
            raise RuntimeError('Finite census resource guard')

    try:
        guard()
        for path, key in ((accepted_path, 'accepted_report_sha256'), (recipe, 'recipe_sha256'), (tokens_path, 'code_tokens_sha256'), (tokenizer_path, 'tokenizer_sha256')):
            if file_sha256(path) != plan[key]:
                raise AssertionError('Same previously audited source bytes required')
        tokens = np.memmap(tokens_path, dtype='<u2', mode='r')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        with recipe.open() as stream:
            for line in stream:
                meta = json.loads(line)
                if meta['domain'] != 'code':
                    continue
                if counts['code_documents'] == 10000:
                    break
                guard()
                if int(meta['group'][:8], 16) % 10 == 0:
                    raise AssertionError('TRAIN-only bucket required')
                counts['code_documents'] += 1
                ids = tokens[meta['source_offset']:meta['source_offset']+meta['source_tokens']].tolist()
                source = tokenizer.decode(ids, skip_special_tokens=False)
                if tokenizer.encode(source, add_special_tokens=False).ids != ids:
                    counts['token_roundtrip_exclusion'] += 1
                    continue
                counts['token_roundtrip_exact'] += 1
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', SyntaxWarning)
                        row = census(source)
                except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
                    counts['bounded_parser_exclusion' if isinstance(error, Unsupported) else 'unparseable_python'] += 1
                    continue
                counts['parseable_bounded_python'] += 1
                counts.update(row)
        guard()
        if file_sha256(__file__) != plan['source_sha256']:
            raise AssertionError('Frozen census source changed')
        result.update(status='COMPLETE_TRAIN_CODE_ASSERTION_CENSUS', counts=dict(counts))
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result.update(wall_seconds=time.time()-start, assigned_chips=0, optimizer_updates=0,
                      model_scores=0, heldout_reads=0, plan_sha256=file_sha256(out/'plan.json'), scope=plan['scope'])
        write_state(out/'report.json', result)
        write_state(PUBLIC/'report.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
