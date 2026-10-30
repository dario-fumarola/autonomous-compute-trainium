"""Bounded TRAIN-only syntax census; never execute or verify source programs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import shutil
import time
import warnings

import numpy as np
from tokenizers import Tokenizer

from compute_trainium.authored_doctest_inventory import inventory
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID = 'authored-doctest-census-v1'
PUBLIC = Path('results/authored_code_targets_20261030/doctest-census-v1')


def run(root):
    started = time.time()
    out = root/ID
    if out.exists() or PUBLIC.exists():
        raise FileExistsError('Preserve every census attempt')
    index = root/'indexed-natural-sources-v1'
    accepted = json.loads((index/'report.json').read_text())
    if accepted['status'] != 'PASS_INDEXED_NATURAL_SOURCES' or accepted['protected_group_overlap']:
        raise AssertionError('Accepted TRAIN sources required')
    recipe = index/'PRIVATE-training-documents.jsonl'
    token_path = root/accepted['source_relative']/'code.u16'
    tokenizer_path = Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    files = ['scripts/inventory_authored_doctests.py', 'compute_trainium/authored_doctest_inventory.py',
             'tests/test_authored_doctest_inventory.py', 'docs/AUTHORED_DOCTEST_CENSUS_20261030.md']
    plan = dict(identity=ID, registered_unix=started, absolute_deadline_unix=started+600,
        maximum_code_documents=10000, maximum_rss_bytes=8*1024**3,
        source_sha256={f:file_sha256(f) for f in files},
        accepted_report_sha256=file_sha256(index/'report.json'),
        recipe_sha256=accepted['files'][recipe.name], code_tokens_sha256=accepted['raw_token_sha256']['code'],
        tokenizer_sha256=file_sha256(tokenizer_path),
        selection='First10000 accepted TRAIN code documents; exact tokenizer roundtrip; '
            'original string-literal doctest prompts and authored integer/boolean outputs. '
            'Conservative syntax coverage only; do not execute or verify source answers.',
        assigned_chips=0, optimizer_updates=0, heldout_reads=0, model_scores=0,
        scientific_fit_admitted=False, terminal_alpha_spent=0)
    out.mkdir(mode=0o700); PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', plan); write_state(PUBLIC/'plan.json', plan)
    counts, groups = Counter(), set()
    result = dict(status='STOPPED_AUTHORED_DOCTEST_CENSUS')

    def guard():
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))
        if (time.time() > plan['absolute_deadline_unix'] or available < 512*1024**3
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > plan['maximum_rss_bytes']
            or min(shutil.disk_usage(root).free, shutil.disk_usage('/').free) < 35*1024**3
            or min(os.statvfs(root).f_favail, os.statvfs('/').f_favail) < 10100):
            raise RuntimeError('Census duration/resource guard')

    try:
        guard()
        if file_sha256(recipe) != plan['recipe_sha256'] or file_sha256(token_path) != plan['code_tokens_sha256']:
            raise AssertionError('Immutable source bytes changed')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokens = np.memmap(token_path, dtype='<u2', mode='r')
        with recipe.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['domain'] != 'code':
                    continue
                if counts['code_documents'] >= plan['maximum_code_documents']:
                    break
                guard()
                if int(row['group'][:8], 16) % 10 == 0:
                    raise AssertionError('TRAIN bucket only')
                counts['code_documents'] += 1
                lo, n = row['source_offset'], row['source_tokens']
                ids = tokens[lo:lo+n].tolist()
                text = tokenizer.decode(ids, skip_special_tokens=False)
                if tokenizer.encode(text, add_special_tokens=False).ids != ids:
                    counts['token_roundtrip_exclusion'] += 1
                    continue
                counts['token_roundtrip_exact'] += 1
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', SyntaxWarning)
                        features = inventory(text)
                except (SyntaxError, ValueError, RecursionError):
                    counts['parser_exclusion'] += 1
                    continue
                counts['bounded_python_documents'] += 1
                counts.update(features)
                if features['unverified_multistep_outputs']:
                    groups.add(row['group'])
                if counts['code_documents'] % 1000 == 0:
                    write_state(out/'progress.json', dict(documents=counts['code_documents'],
                                                        seconds=time.time()-started))
        guard()
        if any(file_sha256(f) != h for f, h in plan['source_sha256'].items()):
            raise AssertionError('Census source changed')
        result.update(status='COMPLETE_AUTHORED_DOCTEST_CENSUS', counts=dict(counts),
                      unverified_multistep_source_groups=len(groups))
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(wall_seconds=time.time()-started, worker_cpu_seconds=usage.ru_utime+usage.ru_stime,
            peak_rss_bytes=usage.ru_maxrss*1024, assigned_chips=0, optimizer_updates=0,
            heldout_reads=0, model_scores=0, plan_sha256=file_sha256(out/'plan.json'),
            scope='Source syntax coverage, not verified examples, training data or an evaluation benchmark.')
        write_state(out/'report.json', result); write_state(PUBLIC/'report.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
