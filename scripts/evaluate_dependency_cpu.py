"""CPU fallback of the already frozen panel;no Native gate is waived."""
import argparse,json,os,time,shutil,resource
from pathlib import Path


def run(a):
 import torch
 from compute_trainium.training import file_sha256
 from compute_trainium.dependency_candidate import DependencyCandidate
 from compute_trainium.dependency_cpu_views import dependency_views
 from compute_trainium.recoverable_dependencies import pack_dependency,real_dependency_view,dependency_metrics
 from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig,rotary_tensors
 from compute_trainium.workspace_cpu_reconstruction import load_streamed_model
 from compute_trainium.stable_target_loss import stable_target_log_probs
 from compute_trainium.execution_isolation import managed_running
 p=json.loads(a.plan.read_text());out=a.root/p['identity'];started=time.time();torch.set_num_threads(8)
 if os.environ.get('USE_TORCH_XLA')!='0' or managed_running():raise RuntimeError('Closed Native owner and CPU-only execution required')
 if file_sha256(a.plan)!=a.plan_sha or any(file_sha256(n)!=h for n,h in p['cpu_source_sha256'].items()):raise AssertionError('Frozen CPU execution source required')
 report=dict(status='STOPPED_DEPENDENCY_CPU_ARM',arm=a.arm,optimizer_updates=0,assigned_chips=0)
 def guard(stage,index=None):
  available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
  if available<512*1024**3 or shutil.disk_usage(a.root).free<35*1024**3 or os.statvfs(a.root).f_favail<10000 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>64*1024**3 or time.time()>p['absolute_deadline_unix']:raise RuntimeError('CPU resource or registered deadline guard')
  (out/f'{a.arm}-progress.json').write_text(json.dumps(dict(stage=stage,index=index,wall_seconds=time.time()-started)))
 try:
  guard('restore_exact_model');cfg=OlmoWorkspaceConfig(**p['model_config'])
  with torch.device('meta'):model=DependencyCandidate(cfg)
  model.to_empty(device='cpu');model.eval().requires_grad_(False)
  if a.arm=='c0':
   if file_sha256(a.root/p['checkpoint_relative'])!=p['checkpoint_sha256']:raise AssertionError('C0 changed')
   state=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model'];model.load_state_dict(state);del state
  else:
   audit=json.loads((a.root/('spliced-workspace-discovery-audit-'+a.arm+'-v1')/'plan.json').read_text());paths=[a.root/'runs'/p['arm_runs'][a.arm]/f'rank{k}-u{p["updates"][a.arm]}.zip' for k in range(64)]
   if [file_sha256(f) for f in paths]!=audit['checkpoint_sha256']:raise AssertionError('Mature audited scientific state changed')
   load_streamed_model(paths,model,label='model')
  versions=tuple(v._version for v in model.parameters());cos,sin=rotary_tensors(cfg,2048);cases=json.loads((a.root/p['panel_relative']).read_text())
  if file_sha256(a.root/p['panel_relative'])!=p['panel_sha256']:raise AssertionError('Frozen panel changed')
  mode='constant' if a.arm=='constant' else 'latent'
  with torch.no_grad():
   if a.arm=='c0':
    records=[]
    for index,reference in enumerate(p['cpu_references']+p['identity_references']):
     guard('independent_HF_gate',index);v=pack_dependency(cases[reference['panel_index']],reference['condition'],cfg.pad_token_id);variant=reference.get('variant','original');r=dependency_views(model,v,cos,sin,mode=mode,pause_id=p['pause_token_id'],variants=[variant])[variant]
     if file_sha256(a.root/reference['relative'])!=reference['sha256']:raise AssertionError('Independent reference changed')
     expected=torch.load(a.root/reference['relative'],weights_only=True,mmap=True)
     if variant=='embedding_replay':expected=dict(expected,mixed_logits=expected['raw_logits'],workspace_nll=expected['raw_nll'])
     rn=-stable_target_log_probs(r['raw_logits'],v['targets']);wn=-stable_target_log_probs(r['mixed_logits'],v['targets']);mask=v['mask'];errors=[float((r['raw_logits']-expected['raw_logits']).abs().max()),float((r['mixed_logits']-expected['mixed_logits']).abs().max()),float((rn.double()[mask]-expected['raw_nll'][mask]).abs().max()),float((wn.double()[mask]-expected['workspace_nll'][mask]).abs().max())]
     records.append(dict(index=index,variant=variant,errors=errors))
     if max(errors[:2])>.001 or max(errors[2:])>.0002:raise AssertionError('CPU independent forward tolerance failed')
    (out/'cpu-forward-gate.json').write_text(json.dumps(dict(status='PASS_INDEPENDENT_CPU_DEPENDENCY_GATE',optimizer_updates=0,records=records),indent=2)+'\n')
   elif not (out/'cpu-forward-gate.json').exists():raise AssertionError('Independent C0 gate precedes all panel scoring')
   rows=[]
   for index,case in enumerate(cases):
    guard('frozen_panel_scoring',index)
    for condition,variants in [('full',['original','strict','strict_rms']),('short',['original'])]:
     v=real_dependency_view(pack_dependency(case,condition,cfg.pad_token_id));results=dependency_views(model,v,cos,sin,mode=mode,pause_id=p['pause_token_id'],variants=variants)
     for variant,r in results.items():
      metrics=dependency_metrics(r['raw_logits'],r['mixed_logits'],v['targets'],v['mask'],r['initial_workspace'],r['workspace_state']);values={k:float(x) for k,x in metrics.items()}
      if not all(__import__('math').isfinite(x) for x in values.values()):raise FloatingPointError('Nonfinite CPU dependency metric')
      rows.append(dict(index=index,id=case['id'],kind=case['kind'],cohort=case['cohort'],group=case['group'],condition=condition,variant=variant,**values))
    if index%16==0:(out/f'PRIVATE-{a.arm}-partial.json').write_text(json.dumps(rows))
  if versions!=tuple(v._version for v in model.parameters()):raise AssertionError('CPU evaluation changed parameters')
  (out/f'{a.arm}-metrics.json').write_text(json.dumps(rows,indent=2)+'\n');report.update(status='COMPLETE_DEPENDENCY_CPU_ARM',case_condition_rows=len(rows),parameters_unchanged=True)
 except BaseException as error:report.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  report.update(wall_seconds=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  (out/f'{a.arm}-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--arm',choices=['c0','ordinary','latent','constant'],required=True);run(p.parse_args())
