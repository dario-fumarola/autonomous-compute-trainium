"""Remove only derived TRAIN feature caches from a completed old branch."""
import json,time,shutil
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.training import file_sha256
ROOT=Path('/run/user/1000/autonomous-compute/instance-nvme')
OUT=Path(__file__).resolve().parent
if managed_running():raise RuntimeError('No retirement during Native execution')
start=time.time();parent=ROOT/'residual-data-volume-v1';p=json.loads((parent/'plan.json').read_text())
folder=ROOT/'runs/residual-data-volume-v1-features'
assert json.loads((folder/'report.json').read_text())['status']=='COMPLETE_FROZEN_RESIDUAL_FEATURES'
assert file_sha256(ROOT/p['checkpoint_relative'])==p['checkpoint_sha256']
required=[ROOT/p['data_relative']/n for n in ['train-tokens.npy','train-order.npy','train-domains.npy','report.json']]
required += [ROOT/r['relative'] for r in p['cpu_references']]
assert all(f.is_file() for f in required)
files=[folder/f'PRIVATE-rank{rank}-train.pt' for rank in range(1,64)]
assert (folder/'PRIVATE-rank0-train.pt').is_file()
rows=[dict(relative=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=file_sha256(f)) for f in files]
receipt=dict(status='REGISTERED_CLOSED_DERIVED_CACHE_RETIREMENT',registered_unix=start,files=rows,
 source_plan_sha256=file_sha256(parent/'plan.json'),extraction_report_sha256=file_sha256(folder/'report.json'),
 retained_checkpoint_sha256=p['checkpoint_sha256'],retained_source_receipts={str(f.relative_to(ROOT)):file_sha256(f) for f in required},
 scope='Ranks1..63 TRAIN features only from closed residual-data-volume branch. Rank0 reference,all source arrays,base and branch models,optimizer/RNG histories,plans,code and aggregate results retained. Cache regeneration requires re-extraction;no byte-identical rerun promised.',
 exact_local_restoration_available=False,scientific_states_deleted=0)
(OUT/'plan.json').write_text(json.dumps(receipt,indent=2)+'\n');before=shutil.disk_usage(ROOT).free
for f,row in zip(files,rows):
 if managed_running() or time.time()-start>600:raise RuntimeError('Bounded inactive-owner requirement')
 assert file_sha256(f)==row['sha256']
 f.with_suffix(f.suffix+'.retired.json').write_text(json.dumps(dict(**row,retirement_receipt='closed-feature-cache-retirement-v1/plan.json',exact_local_restoration_available=False))+'\n')
 f.unlink()
result=dict(status='COMPLETE_CLOSED_DERIVED_CACHE_RETIREMENT',files_retired=len(rows),bytes_retired=sum(r['bytes'] for r in rows),free_before=before,free_after=shutil.disk_usage(ROOT).free,wall_seconds=time.time()-start,scientific_states_deleted=0,source_files_deleted=0,assigned_chips=0)
(OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
