"""Bounded TRAIN-only feasibility audit; no model or held-out source access."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import shutil
import time
import warnings

import numpy as np
from tokenizers import Tokenizer

from compute_trainium.authored_code_assertions import Unsupported
from compute_trainium.authored_doctest_arithmetic import verified_doctest_arithmetic
from compute_trainium.natural_code_context import token_span
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state

ID = 'authored-doctest-arithmetic-train-audit-v1'
PUBLIC = Path('results/authored_doctest_arithmetic_20261029/train-audit-v1')


def run(root):
    start = time.time()
    out = root/ID
    if out.exists() or PUBLIC.exists():
        raise FileExistsError('Preserve prior finite source audits')
    data = root/'indexed-natural-sources-v1'
    accepted = json.loads((data/'report.json').read_text())
    if accepted['status'] != 'PASS_INDEXED_NATURAL_SOURCES' or accepted['protected_group_overlap']:
        raise AssertionError('Previously audited TRAIN-only source required')
    tokenizer_path = Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    recipe = data/'PRIVATE-training-documents.jsonl'
    tokens_path = root/accepted['source_relative']/'code.u16'
    sources = ['scripts/audit_authored_doctest_arithmetic.py', 'compute_trainium/authored_code_assertions.py', 'compute_trainium/authored_doctest_arithmetic.py',
               'compute_trainium/natural_code_context.py', 'tests/test_authored_doctest_arithmetic.py']
    plan = dict(identity=ID, registered_unix=start, absolute_deadline_unix=start+600,
                maximum_code_documents=10000, maximum_rss_bytes=8*1024**3,
                source_sha256={f:file_sha256(f) for f in sources},
                accepted_report_sha256=file_sha256(data/'report.json'),
                recipe_sha256=accepted['files'][recipe.name],
                code_tokens_sha256=accepted['raw_token_sha256']['code'],
                tokenizer_sha256=file_sha256(tokenizer_path),
                selection='First 10000 accepted indexed TRAIN code documents, unchanged source order and tokenizer roundtrip. '
                          'Whole-file bounded Python AST; genuine triple-quoted docstrings with authored interactive prompts, local constant assignments, pure integer expressions and existing canonical integer output lines, '
                          'at module/class/function scope; unknown prompt statements forget all bindings. At least two expression-tree arithmetic '
                          'operations, authentic answer verified and absent from operand literals/dependency-span numeric text. '
                          'All used premises must fit the original contiguous512-source-token prefix. No source execution or model scoring.',
                assigned_chips=0, optimizer_updates=0, heldout_reads=0,
                scope='TRAIN coverage lower bound only. No generated labels, fitted model, evaluation panel or scientific fit admission.')
    out.mkdir(mode=0o700)
    PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', plan)
    write_state(PUBLIC/'plan.json', plan)
    counts = Counter()
    operations = Counter()
    groups = set()
    rows = []
    result = dict(status='STOPPED_AUTHORED_DOCTEST_ARITHMETIC_TRAIN_AUDIT')

    def guard():
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (time.time() > plan['absolute_deadline_unix']
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > plan['maximum_rss_bytes']
                or available < 512*1024**3
                or min(shutil.disk_usage(root).free, shutil.disk_usage('/').free) < 35*1024**3):
            raise RuntimeError('Finite audit resource guard')

    try:
        guard()
        if file_sha256(recipe) != plan['recipe_sha256'] or file_sha256(tokens_path) != plan['code_tokens_sha256']:
            raise AssertionError('Original accepted TRAIN bytes changed')
        tokens = np.memmap(tokens_path, dtype='<u2', mode='r')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        with recipe.open() as stream:
            for line in stream:
                meta = json.loads(line)
                if meta['domain'] != 'code':
                    continue
                if counts['code_documents'] >= plan['maximum_code_documents']:
                    break
                guard()
                if int(meta['group'][:8], 16) % 10 == 0:
                    raise AssertionError('Only the already admitted TRAIN bucket is allowed')
                counts['code_documents'] += 1
                lo, n = meta['source_offset'], meta['source_tokens']
                ids = tokens[lo:lo+n].tolist()
                text = tokenizer.decode(ids, skip_special_tokens=False)
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if encoded.ids != ids:
                    counts['token_roundtrip_exclusion'] += 1
                    continue
                counts['token_roundtrip_exact'] += 1
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', SyntaxWarning)
                        matches = verified_doctest_arithmetic(text)
                except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
                    key = 'bounded_parser_exclusion' if isinstance(error, Unsupported) else 'unparseable_python'
                    counts[key] += 1
                    continue
                counts['parseable_bounded_python'] += 1
                counts['verified_authored_assertions'] += len(matches)
                for match in matches:
                    target = token_span(text, encoded.offsets, match['target_start'], match['target_end'])
                    definition = token_span(text, encoded.offsets, match['premise_start'], match['premise_end'])
                    if target is None or definition is None:
                        counts['token_alignment_exclusion'] += 1
                        continue
                    a, b = target
                    if a < 1 or b-definition[0] > 513:
                        counts['premises_outside_512_source_context'] += 1
                        continue
                    counts['context_eligible_assertions'] += 1
                    counts['authored_answer_tokens'] += b-a
                    operations[str(match['operations'])] += 1
                    groups.add(meta['group'])
                    virtual_target = meta['start']+1+a
                    window_start = ((virtual_target-1)//512)*512
                    packed_available = meta['start']+1+definition[0] >= window_start
                    counts['packed_training_context_contains_premises'] += int(packed_available)
                    rows.append(dict(document_id=meta['id'], group=meta['group'], source_offset=lo,
                                     source_tokens=n, target_tokens=list(target), premise_tokens=list(definition),
                                     operations=match['operations'], packed_premises_available=packed_available))
                if counts['code_documents'] % 1000 == 0:
                    write_state(out/'progress.json', dict(documents=counts['code_documents'], seconds=time.time()-start))
        guard()
        if any(file_sha256(f) != h for f, h in plan['source_sha256'].items()):
            raise AssertionError('Frozen parser changed during audit')
        write_state(out/'PRIVATE-source-spans.json', rows)
        result.update(status='COMPLETE_AUTHORED_DOCTEST_ARITHMETIC_TRAIN_AUDIT', counts=dict(counts),
                      operation_counts=dict(operations), eligible_source_groups=len(groups),
                      private_spans_sha256=file_sha256(out/'PRIVATE-source-spans.json'),
                      limitation='Strict authored-doctest integer grammar, whole-file parsing, truncation, tokenizer alignment and512-token context exclude many authentic dependencies. Operations count an expression tree, including repeated dependencies; this is not proof of runtime execution or model reasoning. '
                                 'Nonmatches do not establish absence of recoverable computation. No model has been evaluated.')
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result.update(wall_seconds=time.time()-start, assigned_chips=0, optimizer_updates=0, heldout_reads=0,
                      model_scores=0, plan_sha256=file_sha256(out/'plan.json'), scope=plan['scope'])
        write_state(out/'report.json', result)
        write_state(PUBLIC/'report.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
