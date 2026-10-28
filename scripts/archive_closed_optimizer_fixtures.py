"""Preserve exact bytes while compressing eight obsolete numerical fixtures."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.exact_gzip_archive import create_archive,restore_archive,sha256_file


def main(root):
    from compute_trainium.execution_isolation import managed_running
    if managed_running():raise RuntimeError('Maintenance requires closed accelerator owners')
    start=time.time();out=root/'probability-workspace-capacity-v1';out.mkdir(exist_ok=False)
    names=['parcae-native-opt-precise-v1-recurrent-primary','parcae-native-opt-precise-v1-ordinary-primary']
    files=[];proofs={}
    for name in names:
        proof=root/'snapshots'/name/'closure.json';report=root/'runs'/name/'report.json'
        if json.loads(proof.read_text())['exit_code']!=0 or json.loads(report.read_text())['status']!='CAPTURED_NATIVE_PARCAE_OPTIMIZER_UPDATES':
            raise AssertionError('Explicit closed numerical fixtures only')
        proofs[name]=dict(closure_sha256=sha256_file(proof),report_sha256=sha256_file(report))
        files += [root/'runs'/name/f for f in ('native-step-1.pt','native-step-2.pt','native-step-3.pt','after-two-full-state.pt')]
    targets={f.resolve() for f in files}
    for base,dirs,entries in os.walk(root):
        if '/snapshots/' in base or '/runtime/' in base:dirs[:]=[];continue
        for name in entries:
            if not name.endswith('.archive.json'):continue
            recipe=Path(base)/name;row=json.loads(recipe.read_text())
            if row.get('parent_relative') and (recipe.parent/row['parent_relative']).resolve() in targets:
                raise AssertionError('A retained archive depends directly on a selected original')
    inventory=[dict(file=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha256_file(f)) for f in files]
    (out/'plan.json').write_text(json.dumps(dict(scope='Eight closed prescribed optimizer-array fixtures; no scientific checkpoints or failed histories',
        files=inventory,proofs=proofs,registered_unix=start,seconds_limit=900,assigned_chips=0),indent=2)+'\n')
    rows=[];before=shutil.disk_usage(root).free;result=dict(status='STOPPED_EXACT_OPTIMIZER_FIXTURE_ARCHIVE')
    try:
        for i,f in enumerate(files):
            # Worst case compressed bytes plus one materialized check and floor.
            needed=f.stat().st_size*(2 if i==0 else 1)+64*1024**2
            if shutil.disk_usage(root).free<35*1024**3+needed or time.time()-start>900:
                raise RuntimeError('Exact archive reserve/time guard')
            recipe,row=create_archive(f)
            if i==0:
                target=out/'roundtrip.pt';restore_archive(recipe,target)
                if sha256_file(target)!=row['original_sha256']:raise AssertionError('Actual restoration differs')
                target.unlink()
            if sha256_file(f)!=row['original_sha256']:raise AssertionError('Original changed')
            f.unlink();rows.append(dict(file=str(f.relative_to(root)),**row))
            (out/'progress.json').write_text(json.dumps(dict(completed=len(rows),records=rows),indent=2)+'\n')
        result['status']='COMPLETE_EXACT_OPTIMIZER_FIXTURE_ARCHIVE'
    except BaseException as e:result.update(error=str(e),error_type=type(e).__name__);raise
    finally:
        result.update(records=rows,wall_seconds=time.time()-start,free_bytes_before=before,free_bytes_after=shutil.disk_usage(root).free,
            scientific_checkpoint_files_deleted=0,original_binary_bytes_exactly_restorable=True,assigned_chips=0,optimizer_updates=0)
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
