"""Private, checksum-verified full states on an existing EBS filesystem."""
import argparse,json,os,shutil,time
from pathlib import Path
import torch
from compute_trainium.training import file_sha256
from scripts.qualify_native import digest


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args()
 a.recovery.mkdir(exist_ok=True);torch.set_num_threads(2);records=[]
 names=['untied-depth-v1-s23-u3456','untied-depth-v1-s23-u3840',
  'stable-cont-v2-ordinary-s23-u10368','stable-cont-v2-recurrent-s23-u10368',
  'stable-cont-v2-ordinary-s23-u11520','stable-cont-v2-recurrent-s23-u11520']
 for name in names:
  source=a.root/'runs'/name/'checkpoint.pt';report=source.with_name('report.json')
  while not report.exists():
   if time.time()>a.deadline_unix:raise TimeoutError('Recovery-copy wait deadline')
   time.sleep(10)
  phase=json.loads(report.read_text())
  if phase['status']!='PASS_PACKED_TRAINING_PHASE':raise ValueError('Source training phase did not pass')
  target=a.recovery/(name+'.pt')
  if not target.exists():
   if shutil.disk_usage(a.recovery).free-source.stat().st_size<35*1024**3:
    raise RuntimeError('Preserve root filesystem reserve; native original retained')
   temporary=target.with_suffix('.copying')
   if temporary.exists():raise FileExistsError('Inspect prior incomplete recovery copy')
   shutil.copy2(source,temporary)
   with temporary.open('rb') as f:os.fsync(f.fileno())
   if file_sha256(source)!=file_sha256(temporary):raise AssertionError('Full recovery copy differs')
   temporary.replace(target)
  checksum=file_sha256(source)
  if checksum!=file_sha256(target):raise AssertionError('Readback copy differs')
  value=torch.load(target,weights_only=False,mmap=True,map_location='cpu')
  if digest({'model':value['model'],'optimizer':value['optimizer']})!=value['state_digest']:
   raise AssertionError('Full recovered model/Adam state differs')
  if value['real_tokens']!=phase['total_real_targets'] or value['cursor']!=phase['cursor'] or 'cpu_rng' not in value:
   raise AssertionError('Recovered cursor, exposure or RNG differs')
  records.append({'run':name,'bytes':target.stat().st_size,'sha256':checksum,'full_state_digest':value['state_digest'],
   'update':value['update'],'real_targets':value['real_tokens'],'cursor':value['cursor'],
   'model_optimizer_rng_schedule_recovered':True,'on_instance_ebs_only':True})
  del value
  (a.recovery/'mechanism-copies-progress.json').write_text(json.dumps(records,indent=2)+'\n')
  print(json.dumps(records[-1]),flush=True)
 receipt={'status':'PASS_SIX_NEW_FULL_STATE_EBS_RECOVERY_COPIES','copies':records,'scope':'Private on-instance EBS copy and full serialized model/Adam digest readback. Native recovery was independently qualified. Not off-instance storage.'}
 (a.recovery/'mechanism-copy-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (a.root/'mechanism-campaign-v2/checkpoint-copy-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
