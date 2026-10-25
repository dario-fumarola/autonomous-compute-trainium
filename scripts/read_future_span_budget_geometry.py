"""Read-only feasibility for a later common1e18-work comparison;no admission."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from compute_trainium.future_span_control_budget import control_branch
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();started=time.time();root=args.root
    identity='future-span-later-budget-readiness-v1';out=root/identity
    public=Path('results/future_span_later_budget_readiness_20261025')
    if out.exists() or public.exists():raise FileExistsError('Preserve an existing geometry audit')
    if any((root/'future-span-work-control-v1/scores').glob('*/report.json')):
        raise ValueError('Fix this possible budget before any current work-control score')
    main_root=root/'future-span-main-v2';file=main_root/'plan.json';plan=json.loads(file.read_text())
    proof_file=root/'future-span-compute-readiness-v1/report.json';proof=json.loads(proof_file.read_text())
    if (proof['status']!='PASS_PROSPECTIVE_WORK_MATCHED_CONTINUATION_GEOMETRY'
            or proof['main_plan_sha256']!=file_sha256(file)):
        raise AssertionError('Original all-row operator-ledger validation required')
    if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):
        raise AssertionError('Original scientific source changed')
    packing=root/plan['data']['packed'];manifest=packing/'manifest.json'
    count_file=packing/'prefix_counts.npy'
    if (file_sha256(manifest)!=plan['data']['manifest_sha256']
            or file_sha256(count_file)!=proof['source_count_prefix_sha256']):
        raise AssertionError('Original source counts changed')
    counts=np.load(count_file,mmap_mode='r');work={};rows={}
    for arm in ['one','recurrent']:
        case=next(c for c in plan['cases'].values() if c['arm']==arm)
        for kind in ['work_table','depth_table']:
            path=main_root/case[kind]
            if file_sha256(path)!=proof['frozen_ledger_sha256'][arm+'-'+kind]:
                raise AssertionError('Independently checked all-row ledger changed')
        work[arm]=np.load(main_root/case['work_table'],mmap_mode='r')
        rows[arm]=len(work[arm])
    candidate=control_branch(work['recurrent'],counts,10**18,origin_update=960)
    exact_budget=candidate['matched_work']['executed_matmuls']
    comparator=control_branch(work['one'],counts,exact_budget,origin_update=960)
    same_data_work=sum(map(int,work['one'][:candidate['total_updates']]))
    same_data=control_branch(work['one'],counts,same_data_work,origin_update=960)
    if (same_data['total_updates']!=candidate['total_updates']
            or max(candidate['total_updates'],comparator['total_updates'])+1>=min(rows.values())
            or (comparator['total_updates']+1)*512>=len(counts)):
        raise AssertionError('Complete source and following-gradient rows must fit')
    result=dict(status='PASS_METADATA_ONLY_LATER_BUDGET_GEOMETRY',identity=identity,
        main_plan_sha256=file_sha256(file),prior_all_row_validation_sha256=file_sha256(proof_file),
        analysis_source_sha256={f:file_sha256(f) for f in ['scripts/read_future_span_budget_geometry.py',
            'compute_trainium/future_span_control_budget.py','compute_trainium/parcae_scientific_schedule.py']},
        common_nominal_work_ceiling=10**18,recurrent_candidate=candidate,one_pass_work_comparator=comparator,
        one_pass_equal_data_comparator=same_data,provisioned_rows=rows,
        source_counts_sha256=file_sha256(count_file),requires_data_or_depth_extension=False,
        selection_disclosure='The possible1e18 ceiling was chosen after completed MAIN discovery and primary-literature review,before any current matched-work score. This script opens only immutable count/work metadata,no model or quality score.',
        checkpoint_adopted=False,training_launched=False,experiment_registered=False,assigned_chips=0,
        model_forwards=0,optimizer_updates=0,test_accessed=False,terminal_alpha_spent=0,
        wall_seconds=time.time()-started,
        scope='Feasibility only. Current controls and prefill qualification retain their exact clocks and priority. A later comparison would need a separate prospective protocol,measured remaining budget,actual stable-state transport/native checks,strong comparators and independent confirmation. No automatic continuation or extension of a current run.')
    out.mkdir();public.mkdir(parents=True);write_state(out/'report.json',result);write_state(public/'report.json',result)
    print(json.dumps({k:result[k] for k in ['status','recurrent_candidate','one_pass_work_comparator']}))


if __name__=='__main__':main()
