import hashlib,json,os,stat,subprocess,time
from pathlib import Path
root=Path('/run/user/1000/autonomous-compute/instance-nvme')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inactive():
 if subprocess.check_output(['docker','ps','-q','--filter','label=ac-trainium-managed=1']).strip():raise RuntimeError('Closed owners only')
inactive();started=time.time();out=root/'probability-closed-probe-sharing-v1';out.mkdir(mode=0o700,exist_ok=False)
canonical=root/'runs/parcae-runtime-scalars-v1-poisson2-warm/probe-initial-states.npy'
expected='1d439cec2b69f9c299232ae81f32f796e6e892ab68813828726508bb159b5e48'
if sha(canonical)!=expected:raise AssertionError('Canonical original hash changed')
info=canonical.stat();candidates=[]
for p in sorted((root/'runs').glob('*/probe-initial-states.npy')):
 if p==canonical or p.is_symlink() or p.parent.is_symlink():continue
 st=p.stat();closure=root/'snapshots'/p.parent.name/'closure.json';report=p.parent/'report.json'
 if not closure.exists() or not report.exists() or json.loads(closure.read_text())['exit_code']!=0:continue
 if not stat.S_ISREG(st.st_mode) or (st.st_size,st.st_uid,st.st_gid,st.st_mode,st.st_dev)!=(info.st_size,info.st_uid,info.st_gid,info.st_mode,info.st_dev):continue
 if st.st_ino==info.st_ino or time.time()-st.st_mtime<24*3600:continue
 candidates.append((p,st,closure,report))
 if len(candidates)==24:break
records=[]
for p,st,closure,report in candidates:
 h=sha(p)
 if h==expected:records.append(dict(path=str(p.relative_to(root)),sha256=h,bytes=st.st_size,inode=st.st_ino,links=st.st_nlink,closure_sha256=sha(closure),report_sha256=sha(report)))
 if time.time()-started>600:raise TimeoutError('Bounded inventory')
plan=dict(registered_unix=started,source_sha256=sha(Path(__file__)),canonical=str(canonical.relative_to(root)),canonical_sha256=expected,records=records,
 scope='Only byte-identical probe-initial-states.npy arrays in successful closed numerical qualifications. Atomic hardlink replacement preserves paths and every byte. Never mutate historical aliases. No scientific checkpoint,optimizer,corpus or receipt removed.')
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');before=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize;count=0
for row in records:
 inactive();p=root/row['path'];s=p.stat()
 if s.st_ino!=row['inode'] or sha(p)!=expected:raise AssertionError('Old array changed before sharing')
 tmp=p.with_name('.probe-initial-states.npy.shared-tmp')
 if tmp.exists():raise FileExistsError('Preserve interrupted transaction')
 os.link(canonical,tmp);os.replace(tmp,p)
 if sha(p)!=expected:raise AssertionError('Shared bytes differ')
 count+=1
 if time.time()-started>900:raise TimeoutError('Bounded maintenance')
for row in records:
 p=root/row['path']
 if sha(root/'snapshots'/p.parent.name/'closure.json')!=row['closure_sha256'] or sha(p.parent/'report.json')!=row['report_sha256']:raise AssertionError('Closed receipts changed')
after=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
r=dict(status='COMPLETE_EXACT_CLOSED_PROBE_ARRAY_SHARING',shared_arrays=count,exact_original_bytes_preserved=True,source_bytes_sha256=expected,
 free_bytes_before=before,free_bytes_after=after,freed_bytes_observed=after-before,scientific_states_removed=0,assigned_chips=0,wall_seconds=time.time()-started,scope=plan['scope'])
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
