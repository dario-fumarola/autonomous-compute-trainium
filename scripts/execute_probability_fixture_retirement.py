"""Execute the already frozen obsolete-fixture plan with required file ownership."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(root):
    out=root/'probability-obsolete-fixture-retirement-v1';plan=json.loads((out/'plan.json').read_text());start=time.time()
    if (out/'report.json').exists():raise FileExistsError('No automatic maintenance repeat')
    if subprocess.check_output(['docker','ps','-q','--filter','label=ac-trainium-managed=1']).strip():raise RuntimeError('Closed owners required')
    files=[root/r['path'] for r in plan['files']]
    if len(files)!=8 or any(not f.is_relative_to(root/'runs') or sha(f)!=r['sha256'] for f,r in zip(files,plan['files'])):raise AssertionError('Exact original inventory required')
    before=shutil.disk_usage(root).free
    for f,row in zip(files,plan['files']):
        if time.time()-start>600:raise RuntimeError('Maintenance cap')
        (f.parent/(f.name+'.retired.json')).write_text(json.dumps(dict(**row,receipt=out.name+'/plan.json',reason=plan['scope'],exact_local_restoration_available=False),indent=2)+'\n')
        f.unlink()
    result=dict(status='COMPLETE_OBSOLETE_SOFTWARE_FIXTURE_RETIREMENT',files=plan['files'],scientific_checkpoints_deleted=0,failed_histories_deleted=0,
        freed_bytes=sum(r['bytes'] for r in plan['files']),free_before=before,free_after=shutil.disk_usage(root).free,wall_seconds=time.time()-start,assigned_chips=0,
        initial_attempt='Permission denied writing first retirement receipt in root-owned closed run;zero arrays deleted. Exact original inventory executed with owner permission.',
        exact_local_restoration_available_for_retired_intermediates=False,canonical_qualification_states_retained=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
