"""Prepare original single-operation outcomes without scoring any model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import shutil
import sqlite3
import time

import numpy as np
from tokenizers import Tokenizer

from compute_trainium.authored_primitive_arithmetic import eligible_cases, choose_panel
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ID = 'primitive-arithmetic-panel-v1'
PUBLIC = Path('results/primitive_arithmetic_20261030/source-v1')


def read(path):
    return json.loads(path.read_text())


def run(root):
    start = time.time(); out = root/ID
    if out.exists() or PUBLIC.exists():
        raise FileExistsError('Preserve every preparation attempt')
    index = root/'indexed-natural-sources-v1'
    accepted, index_plan = read(index/'report.json'), read(index/'plan.json')
    if accepted['status'] != 'PASS_INDEXED_NATURAL_SOURCES' or accepted['protected_group_overlap']:
        raise AssertionError('Accepted indexed sources required')
    source = root/accepted['source_relative']; db = source/'documents.sqlite'
    tokenizer_path = Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    files = ['scripts/prepare_primitive_arithmetic_panel.py', 'compute_trainium/authored_primitive_arithmetic.py',
             'compute_trainium/authored_direct_arithmetic.py', 'compute_trainium/direct_arithmetic_panel.py',
             'compute_trainium/source_arithmetic.py', 'compute_trainium/natural_code_context.py',
             'tests/test_authored_primitive_arithmetic.py', 'docs/PRIMITIVE_ARITHMETIC_DIAGNOSTIC_20261030.md']
    plan = dict(identity=ID, registered_unix=start, absolute_deadline_unix=start+1200,
        maximum_documents=100000, maximum_rss_bytes=8*1024**3,
        source_sha256={f:file_sha256(f) for f in files},
        index_report_sha256=file_sha256(index/'report.json'), database_sha256=index_plan['database_sha256'],
        recipe_sha256=accepted['files']['PRIVATE-training-documents.jsonl'],
        math_tokens_sha256=accepted['raw_token_sha256']['math'], tokenizer_sha256=file_sha256(tokenizer_path),
        excluded_group_files=index_plan['excluded_group_files'], quota_per_operator=64, minimum_per_operator=32,
        source_context=256, maximum_answer_tokens=16, assigned_chips=0, optimizer_updates=0, model_scores=0,
        expression_overlap_policy='Not excluded or measured: basic arithmetic facts may repeat in TRAIN. '
            'This diagnoses elementary performance on distinct source groups, not novel-expression transfer.',
        confirmation_reserves_accessed=False, terminal_alpha_spent=0,
        scope='Exploratory prerequisite diagnostic. Fixed original single-operation integer '
              'equalities and source-only hash selection. No change to the multi-operation primary, '
              'no efficiency claim, no model-dependent selection or scientific fit admission.')
    out.mkdir(mode=0o700); PUBLIC.mkdir(parents=True)
    write_state(out/'plan.json', plan); write_state(PUBLIC/'plan.json', plan)
    counts, candidates, groups = Counter(), [], set()
    result = dict(status='STOPPED_PRIMITIVE_ARITHMETIC_PREPARATION'); connection = None

    def guard():
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))
        if (time.time() > plan['absolute_deadline_unix'] or available < 512*1024**3
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > plan['maximum_rss_bytes']
            or min(shutil.disk_usage(root).free, shutil.disk_usage('/').free) < 35*1024**3):
            raise RuntimeError('Preparation duration/resource guard')

    try:
        guard()
        recipe = index/'PRIVATE-training-documents.jsonl'
        if (file_sha256(db) != plan['database_sha256'] or file_sha256(recipe) != plan['recipe_sha256']
            or file_sha256(source/'math.u16') != plan['math_tokens_sha256']):
            raise AssertionError('Immutable source bytes changed')
        excluded = set()
        for relative, digest in plan['excluded_group_files'].items():
            path = root/relative
            if file_sha256(path) != digest:
                raise AssertionError('Protected source registry changed')
            excluded.update(r['group'] for r in read(path))
        with recipe.open() as stream:
            for line in stream:
                row = json.loads(line); groups.add(row['group'])
        if any(int(g[:8], 16) % 10 == 0 for g in groups):
            raise AssertionError('Training bucket exclusion violated')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokens = np.memmap(source/'math.u16', dtype='<u2', mode='r')
        connection = sqlite3.connect('file:'+str(db)+'?mode=ro', uri=True)
        cursor = connection.execute("select id,source_group,offset,length from documents where domain='math' and split='train' order by id")
        for document, group, lo, length in cursor:
            if int(group[:8], 16) % 10 != 0 or group in excluded:
                continue
            if counts['documents'] >= plan['maximum_documents']:
                break
            guard()
            if group in groups:
                raise AssertionError('Training source overlap')
            counts['documents'] += 1
            n = min(length, 16384); ids = tokens[lo:lo+n].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=False)
            encoding = tokenizer.encode(text, add_special_tokens=False)
            if encoding.ids != ids:
                counts['roundtrip_exclusion'] += 1
                continue
            counts['roundtrip_exact'] += 1
            for row in eligible_cases(text, encoding):
                counts['eligible_'+row['stratum']] += 1
                candidates.append(dict(document_id=document, group=group, source_offset=lo,
                                       source_tokens=n, **row))
            if counts['documents'] % 2000 == 0:
                write_state(out/'progress.json', dict(documents=counts['documents'],
                    candidates=len(candidates), wall_seconds=time.time()-start))
        selected, bins, ready = choose_panel(candidates, plan['quota_per_operator'], plan['minimum_per_operator'])
        if not ready:
            result.update(status='INSUFFICIENT_PRIMITIVE_ARITHMETIC_COVERAGE', counts=dict(counts), bins=bins)
            return
        rows = []
        for i,row in enumerate(selected):
            lo = row['source_offset']; a,b = row['target_tokens']; begin = row['window_start']
            seq = tokens[lo+begin:lo+b].astype(np.int64).tolist()
            shift = int(begin == 0)
            if shift:
                seq = [1]+seq
            first = a-begin+shift
            if first < 1 or len(seq)-1 > 257 or len(seq)-first > 16:
                raise AssertionError('Original prefix/answer bounds')
            rows.append(dict(index=i, **row, tokens=seq, answer_start=first))
        write_state(out/'PRIVATE-panel.json', rows)
        if any(file_sha256(f) != h for f,h in plan['source_sha256'].items()):
            raise AssertionError('Frozen source changed')
        guard()
        result.update(status='FROZEN_PRIMITIVE_ARITHMETIC_DEVELOPMENT_PANEL', counts=dict(counts), bins=bins,
            cases=len(rows), source_groups=len({r['group'] for r in rows}),
            unique_expressions=len({r['expression_key'] for r in rows}),
            answer_tokens=sum(len(r['tokens'])-r['answer_start'] for r in rows),
            single_token_answers=sum(len(r['tokens'])-r['answer_start'] == 1 for r in rows),
            source_group_overlap=0, protected_group_overlap=0,
            files={'PRIVATE-panel.json':file_sha256(out/'PRIVATE-panel.json')})
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        if connection is not None:
            connection.close()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(wall_seconds=time.time()-start, worker_cpu_seconds=usage.ru_utime+usage.ru_stime,
            peak_rss_bytes=usage.ru_maxrss*1024, assigned_chips=0, optimizer_updates=0, model_scores=0,
            protected_groups_scored=0, terminal_alpha_spent=0, plan_sha256=file_sha256(out/'plan.json'),
            scope=plan['scope'], expression_overlap_policy=plan['expression_overlap_policy'])
        write_state(out/'report.json', result); write_state(PUBLIC/'report.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
