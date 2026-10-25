"""Freeze prefix-only context length/noise policy before scientific scoring."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from compute_trainium.code_context_views import views
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/'natural-code-context-v1';public=Path('results/natural_code_context_20261024')
    report_file=out/'scoring-views-v3-report.json'
    if report_file.exists():raise FileExistsError('No silent scoring-view replacement')
    original=json.loads((out/'report.json').read_text())
    if file_sha256(out/'panel.json')!=original['panel_sha256']:raise AssertionError('Original syntax-only selection changed')
    old=json.loads((out/'panel.json').read_text())['records'];rows=[]
    for point in old:
        lo=point['target_start_token'];row={**point,'full_start_token':max(0,lo-1008)}
        row['state_key']=1_000_000_007+lo
        if row['definition_start_token']<row['full_start_token']:raise AssertionError('Fixed prefix removed a selected definition; do not silently exclude it')
        if not 1<=row['target_end_token']-lo<=16:raise AssertionError('Declared identifier length bound changed')
        rows.append(row)
    corpus=a.root/'variable-depth-fresh-v1/dev.json';docs={d['id']:d for d in json.loads(corpus.read_text())}
    for row in rows:views(docs[row['document_id']]['tokens'],row)
    plan={'identity':'natural-code-context-position-only-views-v3','registered_unix':started,
        'selection_panel_sha256':original['panel_sha256'],'full_prefix_tokens':1008,'short_prefix_tokens':128,
        'maximum_identifier_tokens':16,'state_key_policy':'1000000007 + absolute next_source_token_index; excludes document hash,token content,target identifier,length and binding. Available before the prediction in both context conditions.',
        'scope':'Position-only noise-key refinement,before any model scoring; preserve all earlier preparations. Derive exactly the same149 syntax-selected original targets. Fix full prefix length independently of target-span length; reserve16 positions within the1024window. Match retained RoPE/noise positions. No new/excluded target,source rewrite,model-based selection or quality result.',
        'software_sha256':{f:file_sha256(f) for f in ['scripts/prepare_code_context_position_views.py','compute_trainium/code_context_views.py']}}
    write_state(out/'scoring-views-v3-plan.json',plan);write_state(public/'scoring-views-v3-plan.json',plan)
    write_state(out/'scoring-panel-v3.json',{'records':rows})
    result={'status':'FROZEN_PREFIX_ONLY_NATURAL_CODE_CONTEXT_VIEWS','selected_targets':len(rows),
        'original_target_set_unchanged':True,'panel_sha256':file_sha256(out/'scoring-panel-v3.json'),
        'selection_panel_sha256':original['panel_sha256'],'plan_sha256':file_sha256(out/'scoring-views-v3-plan.json'),
        'assigned_chips':0,'optimizer_updates':0,'model_scores_seen':False,'test_accessed':False,
        'wall_seconds':time.time()-started,'scope':plan['scope']}
    write_state(report_file,result);write_state(public/'scoring-views-v3-report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
