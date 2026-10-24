"""Independent full-array and all-generation provenance audit, without scoring."""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from compute_trainium.training import file_sha256
from scripts.prepare_learning_corpus import reserve


def audit_target_prefixes(tokens,domains,counts,windows,guard=lambda:None):
    if tokens.shape!=(windows*1024+1,) or domains.shape!=tokens.shape or counts.shape!=(windows+1,3):
        raise AssertionError('Invalid packed shapes')
    if domains[0]!=0 or np.any(counts[0]) or not 0<=tokens[0]<32000:
        raise AssertionError('Invalid masked starting boundary')
    totals=np.zeros(3,dtype=np.int64)
    for window in range(0,windows,1024):
        guard();end=min(windows,window+1024)
        begin=window*1024+1;stop=end*1024+1
        block=domains[begin:stop].reshape(end-window,1024)
        if block.min()<0 or block.max()>3:raise AssertionError('Unknown domain/mask label')
        local=np.stack([(block==label).sum(1) for label in [1,2,3]],axis=1)
        expected=local.cumsum(0)+totals
        if not np.array_equal(expected,counts[window+1:end+1]):raise AssertionError('Window-prefix target counts differ')
        totals=expected[-1]
        values=tokens[begin:stop]
        if values.min()<0 or values.max()>=32000:raise AssertionError('Token outside pinned vocabulary')
    return totals


def main():
    p=argparse.ArgumentParser();p.add_argument('--generations',nargs='+',type=Path,required=True)
    p.add_argument('--heldout',type=Path,required=True);p.add_argument('--extension',type=Path,required=True)
    a=p.parse_args();started=time.time();reserve(a.extension)
    manifest=json.loads((a.extension/'manifest.json').read_text())
    plan=json.loads((a.extension/'plan.json').read_text())
    if manifest['status']!='PASS_FROZEN_CHAINED_NATURAL_EXTENSION':raise ValueError('Preparation incomplete')
    if file_sha256(a.extension/'plan.json')!=manifest['plan_sha256']:raise AssertionError('Plan changed')
    if [file_sha256(g/'manifest.json') for g in a.generations]!=plan['generation_manifest_sha256']:
        raise AssertionError('Generation membership/order changed')
    for index,generation in enumerate(a.generations):
        old=json.loads((generation/'packed/manifest.json').read_text())
        for name,sha in old['files'].items():
            if file_sha256(generation/'packed'/name)!=sha:raise AssertionError('Prior packed file changed')
        if index:
            before=json.loads((generation/'manifest.json').read_text())
            if before['parent_manifest_sha256']!=file_sha256(a.generations[index-1]/'manifest.json'):
                raise AssertionError('Prior generation lineage differs')
    parent=a.generations[-1]
    new=json.loads((a.extension/'packed/manifest.json').read_text())
    for key in ['tokens','domains','prefix_counts']:
        before=np.load(parent/'packed'/(key+'.npy'),mmap_mode='r')
        after=np.load(a.extension/'packed'/(key+'.npy'),mmap_mode='r')
        if after.dtype!=before.dtype or after.shape[1:]!=before.shape[1:] or len(after)<=len(before):
            raise AssertionError('Invalid appended array')
        for first in range(0,len(before),1<<20):
            reserve(a.extension);end=min(len(before),first+(1<<20))
            if not np.array_equal(before[first:end],after[first:end]):raise AssertionError('Parent prefix changed')
    tokens=np.load(a.extension/'packed/tokens.npy',mmap_mode='r')
    domains=np.load(a.extension/'packed/domains.npy',mmap_mode='r')
    counts=np.load(a.extension/'packed/prefix_counts.npy',mmap_mode='r')
    # Reconstruct every stored window-prefix count, not just the final sum.
    totals=audit_target_prefixes(tokens,domains,counts,new['windows'],lambda:reserve(a.extension))
    expected_totals=np.array([new['domain_counts'][d] for d in ['web','math','code']])
    if not np.array_equal(totals,expected_totals) or new['real_targets']!=int(totals.sum()):
        raise AssertionError('Final manifest counts differ')
    for name,sha in new['files'].items():
        if file_sha256(a.extension/'packed'/name)!=sha:raise AssertionError('New packed checksum differs')
    source=json.loads((a.extension/'source.json').read_text())
    if source['parent_source_sha256']!=file_sha256(parent/'source.json'):raise AssertionError('Source parent changed')
    db=sqlite3.connect(f'file:{a.extension}/extension/documents.sqlite?mode=ro',uri=True)
    protected=set();repeated=0
    for index,generation in enumerate(a.generations):
        path=generation/('documents.sqlite' if index==0 else 'extension/documents.sqlite')
        if file_sha256(path)!=plan['prior_database_sha256'][index]:raise AssertionError('Prior database changed')
        alias=f'prior{index}';db.execute(f'attach database ? as {alias}',(str(path),))
        repeated+=db.execute(f'select count(*) from documents n join {alias}.documents p on n.id=p.id').fetchone()[0]
        protected.update(row[0] for row in db.execute(f"select source_group from {alias}.documents where split!='train'"))
    if repeated:raise AssertionError('New source identity occurs in a previous generation')
    for name,sha in zip(['dev.json','test-sealed.json'],plan['protected_panel_sha256']):
        if file_sha256(a.heldout/name)!=sha or file_sha256(a.extension/name)!=sha:raise AssertionError('Protected panel changed')
        protected.update(d['group'] for d in json.loads((a.heldout/name).read_text()))
    db.execute('create temp table protected(source_group text primary key)')
    db.executemany('insert into protected values(?)',[(g,) for g in protected])
    overlap=db.execute('select count(*) from documents n join protected p on n.source_group=p.source_group').fetchone()[0]
    if overlap or db.execute("select count(*) from documents where split!='train'").fetchone()[0]:
        raise AssertionError('New source crosses protected groups/splits')
    accepted=db.execute('select domain,count(*),sum(length) from documents group by domain').fetchall();db.close()
    if {d:n for d,_,n in accepted}!=manifest['additional_domain_targets']:raise AssertionError('SQLite exposure differs')
    report={'status':'PASS_INDEPENDENT_CHAINED_EXTENSION_AUDIT','manifest_sha256':file_sha256(a.extension/'manifest.json'),
        'generation_count':len(a.generations),'whole_parent_arrays_equal':True,'all_generation_packed_checksums_verified':True,
        'all_window_prefix_counts_independently_reconstructed':True,'real_targets':int(totals.sum()),
        'domain_targets':dict(zip(['web','math','code'],map(int,totals))),'exact_id_overlap':repeated,
        'protected_group_overlap':overlap,'new_documents_by_domain':{d:n for d,n,_ in accepted},'seconds':time.time()-started,
        'scope':'Full parent numeric-prefix equality, vocabulary/masks, every window target prefix, checksums and all-generation SQLite identity/group audit. Does not independently repeat approximate raw-text duplicate screening, score TEST or qualify model-source migration.'}
    (a.extension/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
