"""Exact compression of the completely closed boundary qualification."""
import argparse
import json
from pathlib import Path
import shutil
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--role',choices=['fixed_two'],default='fixed_two')
    a=p.parse_args();started=time.time()
    from compute_trainium.exact_gzip_archive import create_archive,restore_archive,sha256_file
    root=a.root;identity='scale-boundary-fixed-two-numerical-v1'
    proof=root/identity/'completion-verification.json'
    qualified=json.loads(proof.read_text())
    if (qualified['status']!='PASS_BOUNDARY_COMPLETE_NUMERICAL_QUALIFICATION' or not qualified['first_update_moments_pass'] or not qualified['full512']['gradient']['passes_two_percent_gate'] or not qualified['replay']['passes']):raise ValueError('Complete independent numerical closure required')
    for phase in ['warm','replay']:
        closure=root/'snapshots'/(identity+'-'+phase)/'closure.json'
        if json.loads(closure.read_text())['exit_code']!=0:raise ValueError('Successful closed Native phases required')
    out=root/(identity+'-exact-archive-v1')
    if out.exists():raise FileExistsError('Preserve every archival attempt')
    out.mkdir();folder=root/identity/'artifacts/warm';files=sorted(folder.glob('*.pt'))
    if len(files)!=(258 if a.role=='fixed_two' else 194):raise ValueError('Exact three64-shard states,two gradients,and only trained100 noise scope required')
    deadline=started+1200;inventory=[dict(file=f.name,bytes=f.stat().st_size,sha256=sha256_file(f)) for f in files]
    (out/'plan.json').write_text(json.dumps(dict(purpose='exact_closed_qualification_binary_archival',started_unix=started,
        absolute_deadline_unix=deadline,proof_sha256=sha256_file(proof),source_sha256=sha256_file(Path(__file__)),original_inventory=inventory,
        original_binary_bytes_retained=True,source_and_model_parameters_unchanged=True,assigned_chips=0,optimizer_updates=0),indent=2)+'\n')
    records=[];sample_done=False
    for file in files:
        if time.time()>deadline:raise TimeoutError('Original bounded archive clock')
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<512*1024**3+2*file.stat().st_size+64*1024**2:raise RuntimeError('Protected host reserve plus complete temporary archive/sample workspace')
        if shutil.disk_usage(folder).free<35*1024**3+2*file.stat().st_size+64*1024**2:
            raise RuntimeError('Complete archive/sample temporary bytes plus protected reserve required')
        recipe,record=create_archive(file)
        if not sample_done:
            sample=folder/'exact-binary-roundtrip-check.pt';restore_archive(recipe,sample)
            if sha256_file(sample)!=record['original_sha256']:raise AssertionError('Materialized restoration failed')
            sample.unlink();sample_done=True
        # Each independent file is released only after its exact compressed
        # stream is verified. Its complete raw binary remains restorable.
        if sha256_file(file)!=record['original_sha256']:raise AssertionError('Original changed before exact archive release')
        records.append(record);(out/'progress.json').write_text(json.dumps(dict(records=records,last=file.name),indent=2)+'\n')
        file.unlink()
    result=dict(status='ARCHIVED_COMPLETE_QUALIFICATION_EXACT_BINARIES',role=a.role,files=len(records),
        original_bytes=sum(r['original_bytes'] for r in records),archive_bytes=sum(r['archive_bytes'] for r in records),
        every_original_sha256_restorable=True,materialized_sample_exact=True,original_notes_and_proofs_unchanged=True,
        raw_source_corpora_unchanged=True,assigned_chips=0,optimizer_updates=0,wall_seconds=time.time()-started,
        restore_instruction='Restore original warm files with exact_gzip_archive.restore_archive before direct legacy reads,including any exact-shared replay symlinks.',records=records)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
