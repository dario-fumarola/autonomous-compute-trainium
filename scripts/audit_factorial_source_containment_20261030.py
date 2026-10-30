"""Read-only independent containment of enriched TRAIN in the audited prefix."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main(root):
    start=time.time();out=root/'factorial-source-containment-20261030-v1'
    public=Path('results/calculation_exposure_20261030/source-containment-v1')
    if out.exists() or public.exists():raise FileExistsError('Preserve each attempt')
    read=lambda p:json.loads(p.read_text())
    experiment=root/'calculation-exposure-factorial-v1/plan.json';exp=read(experiment)
    selected=root/exp['data_receipts']['enriched']['relative']/'PRIVATE-training-documents.jsonl'
    original=root/exp['data_receipts']['original']['relative']/'PRIVATE-training-documents.jsonl'
    audit=root/'latent-exposure-source-audit-v1/plan.json';a=read(audit)
    plan=dict(registered_unix=start,absolute_deadline_unix=start+120,
        source_sha256=file_sha256(__file__),experiment_plan_sha256=file_sha256(experiment),
        selected_recipe_sha256=file_sha256(selected),original_recipe_sha256=file_sha256(original),
        prior_source_audit_sha256=file_sha256(audit),maximum_audited_virtual_position=a['maximum_virtual_position'],
        fitted_virtual_end=1957*128*512+1,maximum_rss_bytes=4*1024**3,
        scope='Metadata-only verification of every source document used by the registered enriched128M prefix. No evaluation targets, model inference, source text decoding or changes to active training.')
    out.mkdir();public.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    result=dict(status='STOPPED_FACTORIAL_SOURCE_CONTAINMENT')
    def guard():
        if time.time()>plan['absolute_deadline_unix'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>plan['maximum_rss_bytes']:
            raise RuntimeError('Read-only containment resource bound')
    try:
        pending={};counts=Counter();tokens=Counter()
        with selected.open() as stream:
            for line in stream:
                v=json.loads(line)
                if v['start']>=plan['fitted_virtual_end']:break
                n=min(v['source_tokens'],plan['fitted_virtual_end']-v['start']-1)
                if n<=0:continue
                if v['id'] in pending:raise AssertionError('Duplicate enriched TRAIN document')
                pending[v['id']]=dict(domain=v['domain'],group=v['group'],source_offset=v['source_offset'],visible_tokens=n)
                counts[v['domain']]+=1;tokens[v['domain']]+=n
                if len(pending)%10000==0:guard()
        wanted=len(pending);scanned=0
        with original.open() as stream:
            for line in stream:
                v=json.loads(line);scanned+=1
                if v['start']>=plan['maximum_audited_virtual_position']:break
                if scanned%10000==0:guard()
                if v['id'] not in pending:continue
                own=pending.pop(v['id'])
                if any(own[k]!=v[k] for k in ('domain','group','source_offset')):
                    raise AssertionError('Source identity/offset differs from original audited recipe')
                # Match the existing source audit's decoded substring bound,
                # which conservatively includes the final boundary token.
                limit=min(v['source_tokens'],plan['maximum_audited_virtual_position']-v['start'])
                if own['visible_tokens']>limit:raise AssertionError('Enriched source escapes the expression-audited substring')
                if not pending:break
        if pending:raise AssertionError('An enriched TRAIN document was absent from the audited source prefix')
        guard()
        result.update(status='PASS_FACTORIAL_SOURCE_CONTAINMENT',verified_documents=wanted,
            scanned_original_documents=scanned,documents_by_domain=dict(counts),visible_tokens_by_domain=dict(tokens),
            unmatched_documents=0,source_identity_mismatches=0,source_range_escapes=0,
            prior_expression_exclusions_remain_applicable=True)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            assigned_chips=0,optimizer_updates=0,evaluation_targets_read=0,model_scores=0,
            protected_groups_scored=0,plan_sha256=file_sha256(out/'plan.json'),overlaps_active_native_fit=True)
        write_state(out/'report.json',result);write_state(public/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
