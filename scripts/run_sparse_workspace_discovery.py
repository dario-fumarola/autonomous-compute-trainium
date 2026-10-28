"""Three bounded conditions;every mature audit must pass before TUNE scoring."""
import argparse,json,os,shutil,time,math,subprocess
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution
ID='sparse-workspace-discovery-v1';PUBLIC=Path('results/sparse_workspace_discovery_20261028')


def retire_audited_gradient_scratch(root,arm):
 if arm not in ['latent','constant','ordinary']:raise ValueError('Exact registered arm required')
 audit=root/('sparse-workspace-discovery-audit-'+arm+'-v1');proof=json.loads((audit/'report.json').read_text());plan=json.loads((audit/'plan.json').read_text())
 if proof['status']!='PASS_SPARSE_INDEPENDENT_MATURE_CPU_AUDIT':raise AssertionError('Independent all179-gradient proof required before scratch retirement')
 native=root/'runs'/(ID+'-'+arm);paths=[native/f'rank{k}-mature-gradients.pt' for k in range(64)];rows=[]
 for rank,path in enumerate(paths):
  if path.is_symlink() or not path.is_file() or file_sha256(path)!=plan['gradient_sha256'][rank]:raise AssertionError('Retire only exact independently audited derivative scratch')
  rows.append(dict(rank=rank,bytes=path.stat().st_size,sha256=plan['gradient_sha256'][rank]))
 record=dict(status='PREPARED_AUDITED_GRADIENT_SCRATCH_RETIREMENT',arm=arm,files=rows,scope='Only numerical derivative scratch is retired after independent all179-gradient agreement.Full model/AdamW/RNG endpoint and private authentic audit input remain;vectors can be recomputed by zero-update replay.This is retirement,not archival.')
 write_state(root/ID/(arm+'-gradient-scratch-retirement.json'),record)
 subprocess.run(['sudo','-n','rm','--',*[str(path) for path in paths]],check=True)
 record['status']='COMPLETE_AUDITED_GRADIENT_SCRATCH_RETIREMENT';record['retired_bytes']=sum(v['bytes'] for v in rows)
 for d in [root/ID,PUBLIC]:write_state(d/(arm+'-gradient-scratch-retirement.json'),record)


def prepare(root):
 if managed_running():raise RuntimeError('Closed qualified owner required')
 parent=root/'sparse-workspace-warm-v1';warm=json.loads((parent/'report.json').read_text());p=json.loads((parent/'plan.json').read_text())
 if warm['status']!='COMPLETE_SPARSE_WARM_AND_REPLAY_PENDING_MATURE_HF_AUDIT':raise AssertionError('Full warm/replay closure required')
 mature=root/'sparse-workspace-warm-mature-audit-v1/report.json'
 if json.loads(mature.read_text())['status']!='PASS_SPARSE_INDEPENDENT_MATURE_CPU_AUDIT':raise AssertionError('Independent mature HF qualification required')
 if warm['records']['warm']['warm_mean_seconds']>3.:raise RuntimeError('Measured whole-update cost exceeds finite discovery design')
 # Worst-case uncompressed FP32 model+two AdamW moments:3*16.60GiB.
 # At most one6GiB derivative scratch survives between independent audits.
 if shutil.disk_usage(root).free<92*1024**3:raise RuntimeError('50GiB states+6GiB scratch+1GiB compiler allowance+35GiB protected reserve required')
 sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()};sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh')
 worker,host=split_snapshot_sources(sources);projection=project_snapshot_inodes(Path('.'),4)
 if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('All four snapshots and protected inode reserve required')
 from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig
 from compute_trainium.sparse_workspace_training import sparse_matrix_work
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 cfg=OlmoWorkspaceConfig(**p['model_config']);ordinary=sparse_matrix_work(cfg,'ordinary');latent=sparse_matrix_work(cfg,'latent');updates=dict(latent=900,constant=900,ordinary=round(900*latent/ordinary))
 data=WorkspaceViewBatches(root/p['data_relative'],'train');limit=max(updates.values())*64
 if limit>len(data):raise AssertionError('No source wrap/repetition in discovery')
 meta=[json.loads(v) for v in (root/'workspace-source-views-v2/train-views.jsonl').read_text().splitlines()];groups={meta[int(data.order[k])]['group'] for k in range(limit)};audit=next(k for k in range(limit,len(data)) if meta[int(data.order[k])]['group'] not in groups)
 phases=[ID+'-'+arm for arm in ['latent','constant','ordinary']]+[ID+'-evaluation'];start=time.time();base=json.loads((root/'workspace-four-arm-discovery-v3/plan.json').read_text())
 import hashlib
 tune=WorkspaceViewBatches(root/p['data_relative'],'tune');tune_meta=[json.loads(v) for v in (root/'workspace-source-views-v2/tune-views.jsonl').read_text().splitlines()]
 evaluation_groups=[hashlib.sha256(str(tune_meta[int(tune.order[k])]['group']).encode()).hexdigest() for k in range(448)]
 plan=dict(identity=ID,purpose='sparse_workspace_three_arm_discovery',world=64,worker_memory_gib=192,registered_unix=start,seconds_limit=10800,phase_seconds_limit=2400,cpu_phase_seconds_limit=1200,absolute_deadline_unix=start+10800,clock_admission_waiver=True,
 source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,phases),phases={n:dict(mode='evaluation' if n.endswith('-evaluation') else 'fit') for n in phases},inode_projection=projection,artifact_bound_components_gib=dict(full_states=50,one_live_derivative_scratch=6,compiler_and_sources=1,protected_reserve=35),model_config=p['model_config'],checkpoint_relative=p['checkpoint_relative'],checkpoint_sha256=p['checkpoint_sha256'],qualification_plan_sha256=file_sha256(parent/'plan.json'),qualification_report_sha256=file_sha256(parent/'report.json'),mature_qualification_sha256=file_sha256(mature),data_relative=p['data_relative'],data_report_sha256=p['data_report_sha256'],mature_audit_index=audit,updates=updates,lr=1e-5,seed=94611,pause_token_id=base['pause_token_id'],checkpoint_writer_wave_size=2,
 anchors=[127,135,143,151],mix=.5,ordinary_objective='.5 mean raw159 CE+.5 mean anchor4 CE',nominal_operations_per_view=dict(ordinary=ordinary,latent=latent,constant=latent),work_ratio=latent/ordinary,nominal_ordinary_rounding_fraction=(updates['ordinary']*ordinary/(900*latent)-1),evaluation_anchors=448,evaluation_groups=evaluation_groups,evaluation_split='tune',weights=[.6,.2,.2],
 primary='Mean32-natural-token deployed fixed-sparse-policy NLL vs ordinary at matched nominal training matrix work;report all raw32 and anchor4 scores separately.',mechanism='Absolute own-workspace anchor4 benefit vs no workspace,vs matched trained constant-query arm;constant-input same-work ablation is distribution-shifting and insufficient alone.',advance_rule='Single-seed DEVELOPMENT only.A promising mechanism requires positive absolute held-out anchor utility and improvement over the trained constant-query and same-emphasis ordinary controls.Broad useful effect target .01nats;anchor-only practical target .03 with broad deterioration <=.005 are separate hypotheses,not confirmation.Replace neither broad endpoint with a selected winning slice.No more budget or mechanism variants automatically admitted by inconclusive/negative scores.',derivative_scratch_policy='After independent179-gradient PASS and exact file hashes,retire only64 derivative scratch files.Keep full model/AdamW/RNG states,authentic audit inputs,hashes,source and cost receipts.Vectors can be recomputed by zero-update replay;retirement is not archival.',
 scope='Fresh commonC0+freshAdamW;900 direct-gradient sparse and constant-query updates vs923 same-target-emphasis ordinary updates.Nominal training matrix work matched within rounding,not hardware time.Equal rates/tuning opportunity0;short constant-rate learning diagnostic,not converged tuned efficiency endpoint.All qualification/tuning/checkpoint costs separate.No PG,teacher,new parameters,learned allocation,reasoning or random-init claim.All179 mature HF audits before first TUNE score.No gold workspace,answer-aware inference or confirmation data use.')
 out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True)
 for d in [out,PUBLIC]:write_state(d/'plan.json',plan)


def run(root):
 out=root/ID;e=AllChipExecution(root,out/'plan.json',out/'execution');p=e.plan;r=dict(status='STOPPED_SPARSE_WORKSPACE_DISCOVERY',records={},audits={})
 try:
  for arm in ['latent','constant','ordinary']:
   phase=ID+'-'+arm;options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase]),'--arm',arm]
   r['records'][arm]=e.native(phase,'scripts.train_sparse_workspace_discovery',options,'COMPLETE_SPARSE_DISCOVERY_ARM_PENDING_MATURE_HF_AUDIT')
   audit='sparse-workspace-discovery-audit-'+arm+'-v1'
   r['audits'][arm]=e.cpu(audit,'scripts.audit_sparse_workspace_mature',['--root',str(root),'--plan',str(out/'plan.json'),'--arm',arm],root/audit/'report.json','PASS_SPARSE_INDEPENDENT_MATURE_CPU_AUDIT')
   retire_audited_gradient_scratch(root,arm)
   for d in [out,PUBLIC]:write_state(d/'progress.json',r)
  phase=ID+'-evaluation';options=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase])]
  r['evaluation']=e.native(phase,'scripts.evaluate_sparse_workspace_discovery',options,'COMPLETE_SPARSE_WORKSPACE_DISCOVERY_EVALUATION')
  from compute_trainium.sparse_workspace_analysis import analyze_sparse
  r['analysis']=analyze_sparse(root,p)
  for d in [out,PUBLIC]:write_state(d/'analysis.json',r['analysis'])
  r['status']='COMPLETE_SPARSE_WORKSPACE_DISCOVERY'
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  r['observed_completed_optimizer_updates']={}
  for arm in ['latent','constant','ordinary']:
   f=root/'runs'/(ID+'-'+arm)/'updates.json';r['observed_completed_optimizer_updates'][arm]=len(json.loads(f.read_text())) if f.exists() else 0
  r['interrupted_update_may_be_unrecorded']=r['status'].startswith('STOPPED');e.close(out/'execution-costs.json');r['plan_sha256']=e.sha
  for d in [out,PUBLIC]:write_state(d/'report.json',r);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
