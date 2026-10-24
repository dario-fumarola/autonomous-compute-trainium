"""Independent accepted-record and raw token-prefix proof for a capacity reserve."""
import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from compute_trainium.training import file_sha256
from scripts.prepare_learning_corpus import reserve


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--derived',type=Path,required=True);a=p.parse_args();started=time.time()
    reserve(a.derived);plan=json.loads((a.derived/'plan.json').read_text())
    prior=a.attempt/'extension/documents.sqlite'
    if file_sha256(prior)!=plan['failed_attempt_accepted_database_sha256']:
        raise AssertionError('Preserved failed-attempt metadata changed')
    if file_sha256(a.attempt/'failure.json')!=plan['failed_attempt_failure_sha256']:
        raise AssertionError('Original failure record changed')
    audit=json.loads((a.derived/'independent-audit.json').read_text())
    if audit['status']!='PASS_INDEPENDENT_CHAINED_EXTENSION_AUDIT':raise AssertionError('Chain audit missing')
    db=sqlite3.connect(f'file:{a.derived}/extension/documents.sqlite?mode=ro',uri=True)
    db.execute('attach database ? as original',(str(prior),))
    if db.execute('select count(*) from (select * from documents except select * from original.documents)').fetchone()[0]:
        raise AssertionError('Derived metadata not an exact admitted-record subset')
    rows=[]
    for domain,quota in plan['balanced_target_quotas'].items():
        reserve(a.derived)
        count,total=db.execute('select count(*),sum(length) from documents where domain=?',(domain,)).fetchone()
        expected=db.execute('select count(*) from (select * from original.documents where domain=? and offset<? except select * from documents)',(domain,quota)).fetchone()[0]
        extra=db.execute('select count(*) from documents where domain=? and offset>=?',(domain,quota)).fetchone()[0]
        if expected or extra or total!=plan['whole_document_selected_targets'][domain]:
            raise AssertionError('Derived records differ from the declared complete-document prefix')
        destination=a.derived/'extension'/(domain+'.u16');source=a.attempt/'extension'/(domain+'.u16')
        if destination.stat().st_size!=2*total:raise AssertionError('Wrong derived byte count')
        remaining=2*total;digest=hashlib.sha256()
        with source.open('rb') as left,destination.open('rb') as right:
            while remaining:
                length=min(1<<20,remaining);x=left.read(length);y=right.read(length)
                if len(x)!=length or x!=y:raise AssertionError('Accepted token prefix changed')
                digest.update(x);remaining-=length
        rows.append({'domain':domain,'documents':count,'real_targets':total,'entire_token_prefix_equal':True,
                     'token_prefix_sha256':digest.hexdigest()})
    db.close()
    if file_sha256(prior)!=plan['failed_attempt_accepted_database_sha256']:raise AssertionError('Original metadata changed during audit')
    report={'status':'PASS_INDEPENDENT_CAPACITY_SUBSET_PROOF','manifest_sha256':file_sha256(a.derived/'manifest.json'),
            'auditor_sha256':file_sha256(__file__),'exact_record_subset':True,'complete_admission_order_prefixes':True,
            'original_failure_and_accepted_metadata_unchanged':True,'records':rows,'wall_seconds':time.time()-started,
            'scope':'Checks every metadata field through independent SQL set differences and every selected token byte against the preserved failed attempt. Complements the independent all-generation packing/provenance audit. Does not rerun raw lexical near-duplicate screening or qualify a model-source migration.'}
    (a.derived/'subset-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
