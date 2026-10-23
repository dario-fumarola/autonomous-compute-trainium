"""Sanitized incremental accounting for this mechanism campaign only."""
import argparse,hashlib,json
from pathlib import Path
from scripts.summarize_recurrent_costs import union_seconds
from compute_trainium.receipt_work import estimate_matmuls


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 prefixes=('unroll-v1-','untied-','independent-untied-','continuation-v2-','stable-cont-v2-','post-v1-','confirmation-v3-','fast-','fused-amp-')
 rows=[];intervals=[];missing=[];starts=[];ends=[]
 dev_bytes=(a.root.parent/'natural-v2-1050m/dev.json').read_bytes()
 dev_sha=hashlib.sha256(dev_bytes).hexdigest();dev_documents=json.loads(dev_bytes);del dev_bytes
 for lp in sorted((a.root/'snapshots').glob('*/launch.json')):
  name=lp.parent.name
  if not name.startswith(prefixes):continue
  launch=json.loads(lp.read_text());cp=lp.with_name('closure.json')
  if not cp.exists():missing.append(name);continue
  closure=json.loads(cp.read_text());report=a.root/'runs'/name/'report.json'
  v=json.loads(report.read_text()) if report.exists() else {};chips=len(launch['physical_chips']);seconds=closure['wall_seconds']
  starts.append(launch['started_unix']);ends.append(launch['started_unix']+seconds)
  if chips:intervals.append((launch['started_unix'],launch['started_unix']+seconds))
  family='unroll' if name.startswith('unroll') else 'untied' if name.startswith(('untied','independent-untied')) else 'longer' if name.startswith(('continuation','stable-cont')) else 'software_performance' if name.startswith(('fast-','fused-amp-')) else 'confirmation'
  scalar=None
  if v.get('status')=='PASS_CHECKPOINT_DEV_EVALUATION' and name.endswith('-dev'):
   parent=a.root/'runs'/name[:-4]/'report.json'
   if parent.exists():
    scalar=json.loads(parent.read_text())
    if scalar['signature'].get('dev_corpus_sha256')!=dev_sha:raise ValueError('Historical readout source hash differs')
  work=estimate_matmuls(v,scalar_training_report=scalar,dev_documents=dev_documents)
  rows.append({'run':name,'family':family,'exit_code':closure['exit_code'],'report_status':v.get('status'),
   'assigned_chips':chips,'wall_seconds':seconds,'assigned_chip_hours':chips*seconds/3600,
   'updates':v.get('updates',0),'phase_real_targets':v.get('phase_real_targets',0),
   'warm_real_targets_per_second':v.get('warm_global_real_targets_per_second'),
   'parameters':v.get('parameters'),'scored_real_targets':v.get('targets',0),**work})
 cpu=[]
 for folder in ['mechanism-states-v1','document-boundary-v1','mechanism-endpoint-cpu-v1','untied-specialization-v1','core-difference-rank-v1','unroll-development-v1','confirmation-campaign-v3/independent-cpu']:
  path=a.root/folder/'report.json'
  if path.exists():
   v=json.loads(path.read_text());cpu.append({'job':folder,'status':v['status'],'wall_seconds':v.get('wall_seconds',v.get('elapsed_seconds',v.get('wall_seconds_including_waits'))),
    'active_scoring_seconds':v.get('active_scoring_seconds'),'configured_threads':16,'accelerator_chips':0})
 span=max(ends)-min(starts) if starts else 0.
 result={'status':'COMPLETE_INCREMENTAL_RECEIPTS' if not missing else 'PARTIAL_INCREMENTAL_RECEIPTS',
  'runs':rows,'missing_closures':missing,'assigned_chip_hours':sum(r['assigned_chip_hours'] for r in rows),
  'accelerator_active_union_hours':union_seconds(intervals)/3600,
  'active_instance_opportunity_chip_hours':union_seconds(intervals)*16/3600,'cpu_only_jobs':cpu,
  'observed_launch_span_hours':span/3600,'observed_span_opportunity_chip_hours':span*16/3600,
  'hours_between_accelerator_launches_in_span':max(0,span-union_seconds(intervals))/3600,
  'known_optimizer_update_matmul_flops':sum(r['optimizer_update_matmul_flops'] for r in rows),
  'known_training_probe_matmul_flops':sum(r['known_training_probe_matmul_flops'] or 0 for r in rows),
  'known_readout_matmul_flops':sum(r['readout_matmul_flops'] or 0 for r in rows),
  'readouts_without_matmul_metadata':[r['run'] for r in rows if r['readout_matmul_flops'] is None],
  'matmul_scope':'Known completed NN optimizer-update and readout work; numerical native probes separated. Historical scalar DEV window counts are reconstructed from the hash-verified frozen DEV source and checked against recorded document/target counts. Scalar probes reuse the first scored window; sharded probes are additional. Excludes CPU work, optimizer/elementwise kernels, communication, compiler-internal work and unobserved work in failed attempts; these are included in wall-time accounting, not silently estimated as neural matmuls.',
  'scope':'New mechanism campaign and declared follow-up. All closed attempts retained. Native launch time includes compile/setup and recovery. Accelerator-active intervals are unioned before charging the whole node. The separate observed launch span includes CPU work and idle/control gaps between first and last recorded jobs; it is a reservation-cost lower bound, not the full instance bill. CPU work overlaps and is not added again. Reused controls and prior campaigns are historical costs, not charged again. Configured CPU threads are not measured CPU core-hours.'}
 (a.output/'execution-costs.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['runs','cpu_only_jobs']},indent=2))


if __name__=='__main__':main()
