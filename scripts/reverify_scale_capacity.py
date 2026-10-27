"""Reverify existing authentic capacity without rewriting its original receipts."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sqlite3
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deadline-unix',type=float,required=True)
    args=parser.parse_args();started=time.time()
    import numpy as np
    from compute_trainium.training import file_sha256
    from scripts.audit_chained_extension import audit_target_prefixes
    if args.output.exists():raise FileExistsError('Preserve original and new data receipts')
    args.output.mkdir();root=args.root;capacity=root/'natural-append-capacity-v4'
    def guard():
        if time.time()>args.deadline_unix:raise TimeoutError('Original thirty-minute source reverification cap')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3:raise RuntimeError('Protected host reserve plus bounded data-audit memory')
        if min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Protected source audit storage reserve')
    guard();manifest=json.loads((capacity/'manifest.json').read_text());plan=json.loads((capacity/'plan.json').read_text())
    audit=json.loads((capacity/'independent-audit.json').read_text());subset=json.loads((capacity/'subset-audit.json').read_text())
    if (manifest['status']!='PASS_FROZEN_CHAINED_NATURAL_EXTENSION' or manifest['synthetic_generation'] is not False
            or audit['status']!='PASS_INDEPENDENT_CHAINED_EXTENSION_AUDIT' or subset['status']!='PASS_INDEPENDENT_CAPACITY_SUBSET_PROOF'
            or audit['manifest_sha256']!=file_sha256(capacity/'manifest.json')
            or subset['manifest_sha256']!=audit['manifest_sha256'] or manifest['plan_sha256']!=file_sha256(capacity/'plan.json')):
        raise ValueError('Complete immutable historical numeric/provenance/subset receipts required')
    generations=[root.parent/'natural-v2-1050m',root/'natural-append-3250m',root/'natural-append-6500m-v2']
    if [file_sha256(g/'manifest.json') for g in generations]!=plan['generation_manifest_sha256']:
        raise AssertionError('Original source generations changed')
    for index,g in enumerate(generations):
        guard();database=g/('documents.sqlite' if index==0 else 'extension/documents.sqlite')
        if file_sha256(database)!=plan['prior_database_sha256'][index]:raise AssertionError('Original provenance/group database changed')
    original=root/'natural-append-17000m-v3';failed_db=original/'extension/documents.sqlite'
    if (file_sha256(failed_db)!=plan['failed_attempt_accepted_database_sha256']
            or file_sha256(original/'failure.json')!=plan['failed_attempt_failure_sha256']):
        raise AssertionError('Original failed selection/admission evidence changed')
    # Compare full selected metadata to the original accepted whole-document
    # prefixes, including IDs,source groups,split,offset,length and licensing.
    # Both ordered scans use the primary-key index; no large EXCEPT temp file.
    selected=sqlite3.connect('file:'+str(capacity/'extension/documents.sqlite')+'?mode=ro',uri=True)
    before=sqlite3.connect('file:'+str(failed_db)+'?mode=ro',uri=True)
    quota=plan['balanced_target_quotas'];totals={k:0 for k in quota};documents={k:0 for k in quota};count=0
    columns='id,domain,source_group,split,cohort,offset,length,source,license,shard'
    expected=(row for row in before.execute('select '+columns+' from documents order by id') if row[5]<quota[row[1]])
    for left,right in itertools.zip_longest(selected.execute('select '+columns+' from documents order by id'),expected):
        if left!=right or left is None or left[3]!='train':raise AssertionError('Current selected provenance differs from accepted prefix')
        totals[left[1]]+=left[6];documents[left[1]]+=1;count+=1
        if count%8192==0:guard()
    selected.close();before.close()
    if totals!=plan['whole_document_selected_targets']:raise AssertionError('Selected authentic source counts changed')
    packing=json.loads((capacity/'packed/manifest.json').read_text())
    if packing!=manifest['packing']:raise AssertionError('Original packing manifest changed')
    for file,sha in packing['files'].items():
        guard()
        if file_sha256(capacity/'packed'/file)!=sha:raise AssertionError('Whole packed array changed')
    if file_sha256(capacity/'source.json')!=packing['source_sha256']:raise AssertionError('Source manifest changed')
    parent=generations[-1];parent_pack=json.loads((parent/'packed/manifest.json').read_text())
    arrays={k:np.load(capacity/'packed'/(k+'.npy'),mmap_mode='r') for k in ['tokens','domains','prefix_counts']}
    for key in arrays:
        prior=np.load(parent/'packed'/(key+'.npy'),mmap_mode='r')
        for first in range(0,len(prior),1<<20):
            guard()
            end=min(len(prior),first+(1<<20))
            if not np.array_equal(prior[first:end],arrays[key][first:end]):raise AssertionError('Accepted6.5B source prefix changed')
    values=audit_target_prefixes(arrays['tokens'],arrays['domains'],arrays['prefix_counts'],packing['windows'],guard)
    if dict(zip(['web','math','code'],map(int,values)))!=packing['domain_counts']:raise AssertionError('Every-window target totals differ')
    # Only opaque file hashes are used for protected panels; no held-out token
    # targets or quality scores are decoded. The unchanged original full audit
    # supplies its source-group exclusion result on the identical metadata.
    for name,sha in [('dev.json',manifest['dev_sha256']),('test-sealed.json',manifest['test_sha256'])]:
        if file_sha256(capacity/name)!=sha:raise AssertionError('Protected panel fingerprint changed')
    for row in json.loads((root/'raw-source-restoration-v1/plan.json').read_text())['rows']:
        file=original/'raw'/row['file']
        if not file.is_file() or file_sha256(file)!=row['receipt']['sha256']:raise AssertionError('Required original raw dependency not restored')
    result=dict(status='REVERIFIED_AUTHENTIC_SCALE_CAPACITY_NO_MODEL_SOURCE_MIGRATION',real_targets=packing['real_targets'],
        domain_targets=packing['domain_counts'],packing_fingerprint=packing['fingerprint'],source_sha256=packing['source_sha256'],
        packed_manifest_sha256=file_sha256(capacity/'packed/manifest.json'),original_manifest_sha256=file_sha256(capacity/'manifest.json'),
        original_independent_audit_sha256=file_sha256(capacity/'independent-audit.json'),original_subset_audit_sha256=file_sha256(capacity/'subset-audit.json'),
        every_selected_provenance_record_rechecked=count,selected_documents=documents,
        whole_packed_arrays_match_original_hashes=True,every_window_prefix_count_reconstructed=True,
        whole_accepted_6500m_prefix_equal=True,all_original_raw_dependencies_present=True,heldout_targets_decoded=False,
        source_group_exclusion_inherited_from_identical_original_provenance=True,
        near_duplicate_screen_inherited_not_rerun=True,scientific_fit_admitted=False,
        wall_seconds=time.time()-started,assigned_chips=0,optimizer_updates=0,
        original_deadline_unix=args.deadline_unix,
        scope='Reverification of preserved authentic data, not a new filter/selection or fresh panel. Original receipts untouched. A scientific source/endpoint registration is still required.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
