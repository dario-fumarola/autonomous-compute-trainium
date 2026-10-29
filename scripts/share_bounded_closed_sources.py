"""Preserve byte-identical closed snapshot sources while sharing their inodes."""
import argparse,json,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from compute_trainium.closed_source_storage import source_record,source_actions,share_source
from compute_trainium.execution_isolation import managed_running
from compute_trainium.training import file_sha256

def main(root,identity,count,offset=0):
    if type(offset) is not int or not 0<=offset<=512:raise ValueError('Bounded historical snapshot offset required')
    if not identity.startswith('probability-closed-source-') or '/' in identity or not 2<=count<=32:raise ValueError('Bounded closed-source identity/count required')
    if managed_running():raise RuntimeError('No maintenance over active Native owners')
    start=time.time();out=root/identity;out.mkdir(exist_ok=False);snapshots=[]
    for path in (root/'snapshots').iterdir():
        if path.is_symlink() or not (path/'launch.json').is_file() or not (path/'closure.json').is_file():continue
        launch=json.loads((path/'launch.json').read_text());snapshots.append((launch['started_unix'],path,launch))
    ordered=sorted(snapshots,key=lambda x:x[0],reverse=True)
    chosen=ordered[offset:offset+count];jobs=[];proofs={}
    for _,path,launch in chosen:
        proofs[path.name]={n:file_sha256(path/(n+'.json')) for n in ('launch','closure')}
        jobs.extend((path,f,h) for f,h in launch['source_sha256'].items())
    if len(jobs)>30000:raise RuntimeError('30000-source inventory bound')
    with ThreadPoolExecutor(8) as pool:records=list(pool.map(lambda x:source_record(root,*x),jobs))
    actions=source_actions(records)
    (out/'plan.json').write_text(json.dumps(dict(scope='Only immutable closed snapshot source paths;exact original bytes,launch/closure receipts and all scientific states preserved. Live worktree excluded. Never edit hard-linked historical aliases.',source_sha256=file_sha256(__file__),proofs=proofs,records=records,actions=actions,registered_unix=start,seconds_limit=600),indent=2)+'\n')
    before=os.statvfs(root).f_favail;completed=0
    with ThreadPoolExecutor(8) as pool:
        for i in range(0,len(actions),256):
            if managed_running() or time.time()-start>600:raise RuntimeError('Closed-source execution bound')
            completed+=len(list(pool.map(lambda x:share_source(root,x),actions[i:i+256])))
            (out/'progress.json').write_text(json.dumps(dict(completed=completed,seconds=time.time()-start)))
        if not all(pool.map(lambda x:file_sha256(root/x['path'])==x['sha256'],records)):raise AssertionError('Historical source bytes differ')
    for name,receipts in proofs.items():
        if any(file_sha256(root/'snapshots'/name/(n+'.json'))!=h for n,h in receipts.items()):raise AssertionError('Historical receipts differ')
    result=dict(status='COMPLETE_BOUNDED_CLOSED_SOURCE_SHARING',source_paths=len(records),closed_snapshots=len(chosen),shared_paths=completed,free_inodes_before=before,free_inodes_after=os.statvfs(root).f_favail,wall_seconds=time.time()-start,original_source_bytes_changed=0,model_or_optimizer_or_corpus_files_modified=0,assigned_chips=0)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True);p.add_argument('--count',type=int,default=16);p.add_argument('--offset',type=int,default=0);a=p.parse_args();main(a.root,a.identity,a.count,a.offset)
