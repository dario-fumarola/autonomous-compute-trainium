"""Create exact source-span TRAIN views;no fit or student score."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
from tokenizers import Tokenizer

from compute_trainium.natural_chain_views import conclusion_view,deduplicate,normalized_answer
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

IDENTITY='natural-chain-views-v1'
PUBLIC=Path('results/natural_chain_views_20261025')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();started=time.time()
    out=a.root/IDENTITY;out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False)
    parent=a.root/'natural-source-chain-pool-v1';previous=json.loads((parent/'report.json').read_text())
    candidates=parent/'source-span-candidates.json';tok=Path('/run/user/1000/autonomous-compute/data-v1/tokenizer.json')
    if previous['status']!='COMPLETE_NATURAL_SOURCE_CHAIN_POOL_FEASIBILITY' or file_sha256(candidates)!=previous['candidates_sha256']:
        raise AssertionError('Completed immutable source pool required')
    sources=['scripts/prepare_natural_chain_views.py','compute_trainium/natural_chain_views.py','tests/test_natural_chain_views.py','docs/NATURAL_CHAIN_VIEW_PREPARATION_20261025.md']
    plan=dict(identity=IDENTITY,registered_unix=started,source_sha256={p:file_sha256(p) for p in sources},
        parent_report_sha256=file_sha256(parent/'report.json'),source_candidates_sha256=file_sha256(candidates),
        tokenizer_sha256=file_sha256(tok),context=128,bos=1,eos=2,seconds_limit=600,
        scope=Path(sources[-1]).read_text(),assigned_chips=0,research_optimizer_updates=0,test_accessed=False)
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    result=dict(status='STOPPED_NATURAL_CHAIN_VIEW_PREPARATION');counts=Counter();eligible=[];views=[]
    try:
        tokenizer=Tokenizer.from_file(str(tok));records=json.loads(candidates.read_text());unique=deduplicate(records)
        for record in unique:
            available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            if time.time()-started>600 or available<512*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Preparation resource cap')
            try:
                normalized_answer(record['original_source_fragments'][-1])
                options=[conclusion_view(record,tokenizer,k,context=128) for k in range(len(record['original_source_fragments'])-1)]
            except ValueError as error:counts[str(error)]+=1;continue
            eligible.append(record)
            for view in options:
                views.append(dict(document_id=record['id'],group=record['group'],arithmetic_ast_group=record['arithmetic_ast_group'],
                    operation_tree_group=record['operation_tree_group'],visible_steps=view.visible_steps,
                    original_operations=record['source_spans']['original_operations'],prompt=view.prompt,target=view.target,
                    inputs=view.inputs,targets=view.targets,answer_mask=view.target_mask,termination_mask=view.termination_mask,
                    answer_ids=view.answer_ids,prefix_ids=view.prefix_ids,answer_start=view.answer_start,
                    original_utf8_sha256=record['original_utf8_sha256'],original_source_spans=record['source_spans']))
        write_state(out/'eligible-source-records.json',eligible);write_state(out/'training-views.json',views)
        if any(file_sha256(p)!=h for p,h in plan['source_sha256'].items()):raise AssertionError('Frozen view logic changed')
        result=dict(status='COMPLETE_TRAIN_ONLY_ORIGINAL_CONCLUSION_VIEWS',source_candidates=len(records),
            unique_arithmetic_trees=len(unique),eligible_problems=len(eligible),source_groups=len({r['group'] for r in eligible}),
            operation_tree_groups=len({r['operation_tree_group'] for r in eligible}),training_views=len(views),
            exclusion_counts=dict(counts),operation_counts=dict(Counter(r['source_spans']['original_operations'] for r in eligible)),
            all_targets_original_substrings=True,no_calculator_labels=True,hidden_work_absent=True,student_scores_read=False,
            views_sha256=file_sha256(out/'training-views.json'),eligible_records_sha256=file_sha256(out/'eligible-source-records.json'),
            scientific_fit_admitted=False)
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(plan_sha256=file_sha256(out/'plan.json'),wall_seconds=time.time()-started,assigned_chips=0,research_optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
