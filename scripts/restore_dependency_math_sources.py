"""Restore accepted immutable source bytes;never creates examples or labels."""
import argparse,concurrent.futures,json,time,os,shutil,hashlib
from pathlib import Path
import requests
from compute_trainium.training import file_sha256


def run(root,domain='math'):
 out=root/('dependency-source-restore-v1' if domain=='math' else 'dependency-code-source-restore-v1');out.mkdir(mode=0o700);started=time.time();raw=json.loads((root/'natural-append-capacity-v4/raw-source-map.json').read_text());records=[x for x in raw if x['domain']==domain]
 if domain=='math':indices=list(range(24))
 else:
  wanted={x['document_id'] for x in json.loads((root/'natural-code-context-v1/panel.json').read_text())['records']};meta=json.loads((root/'variable-depth-fresh-v1/dev.json').read_text());indices=sorted({x['shard'] for x in meta if x['id'] in wanted})
 plan=dict(identity=out.name,registered_unix=started,absolute_deadline_unix=started+1800,maximum_readers=4,maximum_new_bytes=9*1024**3,optimizer_updates=0,assigned_chips=0,inputs=[dict(index=k,**x['receipt']) for k,x in enumerate(records) if k in indices],scope='Reacquire original accepted immutable math parquet bytes from pinned public dataset revisions only.Exact SHA256 required.No model or source-example selection,TEST access or changed source corpus.')
 (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');results=[]
 def task(k,x):
  receipt=x['receipt'];existing=Path(x['path'])
  def guard():
   available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
   if available<512*1024**3 or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3 or time.time()>plan['absolute_deadline_unix']:raise RuntimeError('Source restoration reserve/time guard')
  guard()
  if existing.is_file():
   if file_sha256(existing)!=receipt['sha256']:raise AssertionError('Existing source SHA mismatch')
   return dict(index=k,path=str(existing),bytes=existing.stat().st_size,sha256=receipt['sha256'],restored=False)
  target=out/(f'math-{k}.parquet' if domain=='math' else f'code-{k}.json.gz');temporary=target.with_suffix('.incomplete');url=f"https://huggingface.co/datasets/{receipt['repository']}/resolve/{receipt['revision']}/{receipt['file']}?download=true";digest=hashlib.sha256();total=0
  response=requests.get(url,stream=True,timeout=(20,90));response.raise_for_status()
  with temporary.open('xb') as f:
   for block in response.iter_content(1024**2):
    guard();f.write(block);digest.update(block);total+=len(block)
    if total>500*1024**2:raise RuntimeError('Pinned shard exceeds finite restoration allowance')
   f.flush();os.fsync(f.fileno())
  if digest.hexdigest()!=receipt['sha256']:raise AssertionError('Restored source SHA mismatch')
  temporary.rename(target);return dict(index=k,path=str(target),bytes=total,sha256=digest.hexdigest(),restored=True)
 result=dict(status='STOPPED_PINNED_DEPENDENCY_SOURCE_RESTORATION',records=[])
 try:
  with concurrent.futures.ThreadPoolExecutor(4) as pool:
   for future in concurrent.futures.as_completed([pool.submit(task,k,x) for k,x in enumerate(records) if k in indices]):
    row=future.result();results.append(row);(out/'progress.json').write_text(json.dumps(dict(files=len(results),restored=sum(v['restored'] for v in results),wall_seconds=time.time()-started)));print('Verified source shard',row['index'],'new',row['restored'],flush=True)
  result.update(status='PASS_PINNED_DEPENDENCY_SOURCE_RESTORATION',records=sorted(results,key=lambda x:x['index']),restored_bytes=sum(x['bytes'] for x in results if x['restored']))
 except BaseException as error:result.update(error_type=type(error).__name__,error='Pinned source restoration stopped;retain private initiating trace and partial files.');raise
 finally:
  result.update(wall_seconds=time.time()-started,optimizer_updates=0,assigned_chips=0);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--domain',choices=['math','code'],default='math');a=p.parse_args();run(a.root,a.domain)
