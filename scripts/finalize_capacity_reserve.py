"""Derive a balanced reserve from a preserved, code-capacity-limited attempt.

Never relax source filters or rewrite the failed attempt. Select whole-document
prefixes of accepted records, independent of any model loss, then reuse packing.
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
import time
from pathlib import Path

from compute_trainium.training import file_sha256
from scripts.prepare_chained_extension import lineage
from scripts.prepare_extension_corpus import concatenate
from scripts.prepare_learning_corpus import freeze_packing,reserve


def capacity_selection(db):
    counts=dict(db.execute('select domain,sum(length) from documents group by domain'))
    if set(counts)!={'web','math','code'}:raise ValueError('All three accepted domains required')
    unit=min(counts['web']//3,counts['math'],counts['code'])
    if unit<=0:raise ValueError('No balanced reserve available')
    quotas={'web':3*unit,'math':unit,'code':unit};selected={}
    for domain,quota in quotas.items():
        low,high,total=db.execute('select min(offset),max(offset+length),sum(length) from documents where domain=? and offset<?',(domain,quota)).fetchone()
        if low!=0 or high!=total or not quota<=total<quota+8192:
            raise ValueError('Accepted prefix has holes or violates whole-document quota')
        selected[domain]=total
    return counts,quotas,selected


def copy_prefix(source,destination,length):
    remaining=length;digest=hashlib.sha256()
    with source.open('rb') as reader,destination.open('wb') as writer:
        while remaining:
            data=reader.read(min(1<<20,remaining))
            if not data:raise ValueError('Accepted token file shorter than recorded prefix')
            writer.write(data);digest.update(data);remaining-=len(data)
    if file_sha256(destination)!=digest.hexdigest():raise AssertionError('Copied token prefix differs')
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--generations',nargs='+',type=Path,required=True);p.add_argument('--heldout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);started=time.time()
    def guard():
        reserve(a.output)
        if time.time()>a.deadline_unix:raise TimeoutError('Capacity finalization deadline')
    try:
        guard();manifests,databases=lineage(a.generations);parent=a.generations[-1]
        failure=json.loads((a.attempt/'failure.json').read_text());original_plan=json.loads((a.attempt/'plan.json').read_text())
        if not failure['error'].startswith("ValueError('Insufficient natural code:"):
            raise ValueError('Only the verified code-capacity exhaustion is eligible')
        if (a.attempt/'manifest.json').exists():raise ValueError('Do not rewrite a successful corpus')
        if any(file_sha256(f)!=h for f,h in original_plan['source_sha256'].items()):
            raise ValueError('Original preparation source changed')
        events=[]
        log=a.attempt.parent/'logs/chained-reserve-v3-prepare_chained_extension.log'
        for line in log.read_text().splitlines():
            try:events.append(json.loads(line))
            except ValueError:pass
        prior=next(x for x in events if x['event']=='prior_sets')
        rebuilt=[x for x in events if x['event']=='index_rebuilt']
        if len(rebuilt)!=3 or rebuilt[-1]['documents']!=prior['documents']:
            raise ValueError('Prior raw duplicate reconstruction was incomplete')
        original_db=a.attempt/'extension/documents.sqlite';original_sha=file_sha256(original_db)
        source=sqlite3.connect(f'file:{original_db}?mode=ro',uri=True)
        counts,quotas,selected=capacity_selection(source)
        if not counts['code']<original_plan['additional_target_quota']//5:raise ValueError('Not a capacity-limited attempt')
        for domain,count in counts.items():
            if (a.attempt/'extension'/(domain+'.u16')).stat().st_size!=2*count:raise ValueError('Token-file exposure differs')
        source.close()
        plan={'identity':'capacity-balanced-natural-reserve-v4','purpose':'DATA_FINALIZATION_ONLY_NO_TRAINING',
            'started_unix':started,'deadline_unix':a.deadline_unix,'synthetic_generation':False,
            'parent_manifest_sha256':file_sha256(parent/'manifest.json'),
            'generation_manifest_sha256':[file_sha256(g/'manifest.json') for g in a.generations],
            'prior_database_sha256':[file_sha256(d) for d in databases],
            'protected_panel_sha256':[file_sha256(a.heldout/n) for n in ['dev.json','test-sealed.json']],
            'tokenizer_sha256':original_plan['tokenizer_sha256'],
            'failed_attempt_plan_sha256':file_sha256(a.attempt/'plan.json'),
            'failed_attempt_failure_sha256':file_sha256(a.attempt/'failure.json'),
            'failed_attempt_accepted_database_sha256':original_sha,
            'available_accepted_targets':counts,'balanced_target_quotas':quotas,'whole_document_selected_targets':selected,
            'selection':'Let unit=min(web//3,math,code). Keep original accepted admission-order whole-document prefixes with starting offset below3*unit/unit/unit. No model loss, new filter or source admission.',
            'source_sha256':{f:file_sha256(f) for f in ['scripts/finalize_capacity_reserve.py','scripts/audit_chained_extension.py','scripts/prepare_chained_extension.py','scripts/prepare_extension_corpus.py','scripts/prepare_learning_corpus.py']}}
        (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');extension=a.output/'extension';extension.mkdir()
        shutil.copyfile(original_db,extension/'documents.sqlite');db=sqlite3.connect(extension/'documents.sqlite')
        copied={}
        for domain,quota in quotas.items():
            guard();db.execute('delete from documents where domain=? and offset>=?',(domain,quota));db.commit()
            copied[domain]=copy_prefix(a.attempt/'extension'/(domain+'.u16'),extension/(domain+'.u16'),selected[domain]*2)
        if dict(db.execute('select domain,sum(length) from documents group by domain'))!=selected:raise AssertionError('Selected metadata differs')
        guard();extension_packing=freeze_packing(extension,db,selected,1024);db.close()
        guard();packing=concatenate(parent,extension,a.output);guard()
        if file_sha256(original_db)!=original_sha:raise AssertionError('Failed-attempt evidence changed')
        for name in ['dev.json','test-sealed.json']:shutil.copy2(a.heldout/name,a.output/name)
        raw=json.loads((a.attempt/'raw-source-map.json').read_text())
        (a.output/'raw-source-map.json').write_text(json.dumps(raw,indent=2)+'\n')
        sources={domain:[x['receipt'] for x in raw if x['domain']==domain] for domain in ['web','math','code']}
        report={'status':'PASS_FROZEN_CHAINED_NATURAL_EXTENSION','synthetic_generation':False,
            'plan_sha256':file_sha256(a.output/'plan.json'),'parent_manifest_sha256':plan['parent_manifest_sha256'],
            'sources':sources,'prior_and_protected_documents_reconstructed':prior['documents'],'duplicate_reconstruction_complete':True,
            'protected_source_groups':prior['protected_groups'],'additional_domain_targets':selected,
            'accepted_token_prefix_sha256':copied,'near_duplicate_audit':manifests[0]['near_duplicate_audit'],
            'packing':packing,'extension_packing':extension_packing,
            'dev_sha256':file_sha256(a.output/'dev.json'),'test_sha256':file_sha256(a.output/'test-sealed.json'),
            'preparation_wall_seconds':time.time()-started,
            'scope':'New identity derived from the preserved failed capacity attempt, with the same source filters and approximately60/20/20 weights. Whole accepted-document prefixes only. Raw lexical duplicate screening is inherited from that attempt, not rerun. Original failed evidence and raw sources remain indispensable private dependencies. Numeric/provenance and subset audits remain required; no model-source migration or new training.'}
        (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'real_targets':packing['real_targets'],'selected_additional_targets':selected,'seconds':report['preparation_wall_seconds']}),flush=True)
    except Exception as error:
        (a.output/'failure.json').write_text(json.dumps({'status':'STOPPED_CAPACITY_FINALIZATION','error_type':type(error).__name__,'seconds':time.time()-started},indent=2)+'\n');raise


if __name__=='__main__':main()
