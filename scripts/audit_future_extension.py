"""Independent whole-prefix/mask and SQLite provenance audit; never scores TEST."""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from compute_trainium.training import file_sha256
from scripts.prepare_learning_corpus import reserve


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--legacy',type=Path,required=True);p.add_argument('--heldout',type=Path,required=True)
    p.add_argument('--extension',type=Path,required=True);a=p.parse_args();started=time.time()
    reserve(a.extension);manifest=json.loads((a.extension/'manifest.json').read_text())
    if manifest['status']!='PASS_FROZEN_SECOND_NATURAL_EXTENSION':raise ValueError('Preparation not complete')
    old=json.loads((a.parent/'packed/manifest.json').read_text())
    new=json.loads((a.extension/'packed/manifest.json').read_text())
    for key in ['tokens','domains','prefix_counts']:
        before=np.load(a.parent/'packed'/(key+'.npy'),mmap_mode='r')
        after=np.load(a.extension/'packed'/(key+'.npy'),mmap_mode='r')
        if after.dtype!=before.dtype or after.shape[1:]!=before.shape[1:] or len(after)<=len(before):
            raise AssertionError('Invalid appended array shape/dtype')
        for first in range(0,len(before),1<<20):
            end=min(len(before),first+(1<<20))
            if not np.array_equal(before[first:end],after[first:end]):raise AssertionError('Original prefix changed')
    tokens=np.load(a.extension/'packed/tokens.npy',mmap_mode='r')
    domains=np.load(a.extension/'packed/domains.npy',mmap_mode='r')
    counts=np.load(a.extension/'packed/prefix_counts.npy',mmap_mode='r')
    totals=np.zeros(4,dtype=np.int64)
    for first in range(0,len(domains),1<<20):
        block=domains[first:first+(1<<20)]
        if block.max()>3:raise AssertionError('Unknown domain/mask label')
        totals+=np.bincount(block,minlength=4)
        values=tokens[first:first+(1<<20)]
        if values.min()<0 or values.max()>=32000:raise AssertionError('Token outside pinned vocabulary')
    if domains[0]!=0 or not np.array_equal(totals[1:],counts[-1]):raise AssertionError('Real-target totals differ')
    expected=np.array([new['domain_counts'][d] for d in ['web','math','code']])
    if not np.array_equal(expected,counts[-1]) or new['real_targets']!=int(expected.sum()):
        raise AssertionError('Manifest target counts differ')
    for name,sha in new['files'].items():
        if file_sha256(a.extension/'packed'/name)!=sha:raise AssertionError('Packed checksum differs')
    for name,sha in old['files'].items():
        if file_sha256(a.parent/'packed'/name)!=sha:raise AssertionError('Original packed checksum changed')
    source=json.loads((a.extension/'source.json').read_text())
    if source['parent_source_sha256']!=file_sha256(a.parent/'source.json'):
        raise AssertionError('Source lineage differs')
    db=sqlite3.connect(f'file:{a.extension}/extension/documents.sqlite?mode=ro',uri=True)
    db.execute('attach database ? as first',(str(a.legacy/'documents.sqlite'),))
    db.execute('attach database ? as second',(str(a.parent/'extension/documents.sqlite'),))
    repeated=sum(db.execute(f'select count(*) from documents n join {name}.documents p on n.id=p.id').fetchone()[0]
                 for name in ['first','second'])
    if repeated:raise AssertionError('Repeated exact source identity')
    protected={row[0] for row in db.execute("select source_group from first.documents where split!='train'")}
    for name in ['dev.json','test-sealed.json']:
        # Identity/group metadata only; no model consumes or scores the panel.
        protected.update(d['group'] for d in json.loads((a.heldout/name).read_text()))
        if file_sha256(a.heldout/name)!=file_sha256(a.extension/name):raise AssertionError('Protected panel changed')
    db.execute('create temp table protected(source_group text primary key)')
    db.executemany('insert into protected values(?)',[(g,) for g in protected])
    overlap=db.execute('select count(*) from documents n join protected p on n.source_group=p.source_group').fetchone()[0]
    if overlap or db.execute("select count(*) from documents where split!='train'").fetchone()[0]:
        raise AssertionError('New TRAIN contains protected group/split')
    accepted=db.execute('select domain,count(*),sum(length) from documents group by domain').fetchall();db.close()
    expected_new=manifest['additional_domain_targets']
    if {d:n for d,_,n in accepted}!=expected_new:raise AssertionError('SQLite/token exposure differs')
    report={'status':'PASS_INDEPENDENT_SECOND_EXTENSION_AUDIT','manifest_sha256':file_sha256(a.extension/'manifest.json'),
            'whole_parent_arrays_equal':True,'all_new_and_parent_packing_checksums_verified':True,
            'real_targets':int(expected.sum()),'domain_targets':dict(zip(['web','math','code'],map(int,expected))),
            'exact_id_overlap':repeated,'protected_group_overlap':overlap,
            'new_documents_by_domain':{d:count for d,count,_ in accepted},'seconds':time.time()-started,
            'scope':'Independent full numeric prefix equality, whole-array vocabulary/mask totals, file checksums and SQLite exact/group isolation. Does not independently repeat approximate raw-text near-duplicate screening, score TEST or qualify a model-source migration.'}
    (a.extension/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
