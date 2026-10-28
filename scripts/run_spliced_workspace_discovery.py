"""One finite uniform-NTP three-arm experiment;all audits before first score."""
import argparse,json,os,shutil,time,subprocess,hashlib,io,zipfile
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution
ID='spliced-workspace-discovery-v1';PUBLIC=Path('results/spliced_workspace_discovery_20261028')


def prepare(root):
 import torch
 from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig
 from compute_trainium.spliced_workspace_training import spliced_matrix_work
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 if managed_running():raise RuntimeError('Closed qualified owner required')
 parent=root/'spliced-workspace-warm-v2';p=json.loads((parent/'plan.json').read_text());warm=json.loads((parent/'report.json').read_text())
 if warm['status']!='COMPLETE_SPARSE_WARM_AND_REPLAY_PENDING_MATURE_HF_AUDIT':raise AssertionError('Complete101 and exact fresh replay required')
 mature=root/'spliced-workspace-warm-v2-mature-audit/report.json'
 if json.loads(mature.read_text())['status']!='PASS_SPLICED_INDEPENDENT_MATURE_CPU_AUDIT':raise AssertionError('Independent mature all179 gate required')
 if warm['records']['warm']['warm_mean_seconds']>3:raise RuntimeError('Measured whole-update cost exceeds bounded design')
 constant=json.loads((root/'spliced-workspace-constant-cpu-reference-v1/report.json').read_text())
 if constant['status']!='PASS_SPLICED_FULL_SIZE_CPU_REFERENCE':raise AssertionError('Independent full-size constant oracle required')
 if shutil.disk_usage(root).free<70*1024**3:raise RuntimeError('35GiB floor plus35GiB maximum incremental states/scratch under explicit retirement schedule required')
 sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()};sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh')
 worker,host=split_snapshot_sources(sources);projection=project_snapshot_inodes(Path('.'),4)
 if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('All four snapshots and protected inode reserve required')
 cfg=OlmoWorkspaceConfig(**p['model_config']);ordinary=spliced_matrix_work(cfg,query_count=0);latent=spliced_matrix_work(cfg);updates=dict(latent=800,constant=800,ordinary=round(800*latent/ordinary));prefixes={}
 for arm,run,count in [('latent','spliced-workspace-warm-v2-warm',101),('ordinary','workspace-four-arm-discovery-v1-ordinary',931)]:
  path=root/'runs'/run
  with zipfile.ZipFile(path/f'rank0-u{count}.zip') as z:receipt=torch.load(io.BytesIO(z.read('metadata.pt')),weights_only=True)
  old=receipt['metadata'];assert old['update']==count and old['data_cursor']==count*64 and len(receipt['order'])==716 and not old['scorer_present']
  prefixes[arm]=dict(updates=count,run_relative='runs/'+run,checkpoint_sha256=[file_sha256(path/f'rank{k}-u{count}.zip') for k in range(64)],metadata_plan_sha256=old['plan_sha256'])
  if arm=='latent':prefixes[arm]['mature_audit_index']=p['mature_audit_index']
 prefixes['constant']=dict(updates=0)
 oldplan=json.loads((root/'workspace-four-arm-discovery-v1/plan.json').read_text())
 if oldplan['checkpoint_sha256']!=p['checkpoint_sha256'] or oldplan['data_report_sha256']!=p['data_report_sha256'] or oldplan['lr']!=1e-5:raise AssertionError('Reused plain-NTP prefix must share exact C0,data and optimizer recipe')
 data=WorkspaceViewBatches(root/p['data_relative'],'train');limit=max(updates.values())*64
 if limit>len(data):raise AssertionError('No source wrap')
 meta=[json.loads(v) for v in (root/'workspace-source-views-v2/train-views.jsonl').read_text().splitlines()];groups={meta[int(data.order[k])]['group'] for k in range(limit)};audit=next(k for k in range(limit,len(data)) if meta[int(data.order[k])]['group'] not in groups)
 for n,h in json.loads((root/p['data_relative']/'report.json').read_text())['records']['train']['files'].items():
  if file_sha256(root/p['data_relative']/n)!=h:raise AssertionError('Frozen authentic TRAIN arrays changed')
 tune=WorkspaceViewBatches(root/p['data_relative'],'tune');tm=[json.loads(v) for v in (root/'workspace-source-views-v2/tune-views.jsonl').read_text().splitlines()];evalgroups=[hashlib.sha256(str(tm[int(tune.order[k])]['group']).encode()).hexdigest() for k in range(448)]
 phases=[ID+'-'+arm for arm in ['latent','constant','ordinary']]+[ID+'-evaluation'];start=time.time();base=json.loads((root/'workspace-four-arm-discovery-v3/plan.json').read_text())
 plan=dict(identity=ID,purpose='spliced_workspace_three_arm_discovery',world=64,worker_memory_gib=192,registered_unix=start,seconds_limit=10800,phase_seconds_limit=2400,cpu_phase_seconds_limit=1200,absolute_deadline_unix=start+10800,clock_admission_waiver=True,
 source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,phases),phases={n:dict(mode='evaluation' if n.endswith('-evaluation') else 'fit') for n in phases},inode_projection=projection,
 model_config=p['model_config'],checkpoint_relative=p['checkpoint_relative'],checkpoint_sha256=p['checkpoint_sha256'],qualification_report_sha256=file_sha256(parent/'report.json'),mature_qualification_sha256=file_sha256(mature),data_relative=p['data_relative'],data_report_sha256=p['data_report_sha256'],mature_audit_index=audit,mature_anchors=list(range(31,159,4)),updates=updates,prefixes=prefixes,arm_runs={arm:ID+'-'+arm for arm in updates},lr=1e-5,sample_seed_base=9821,pause_token_id=base['pause_token_id'],query_count=32,eligible_start=31,mix=.5,
 constant_reference={n:constant[n] for n in ['gradient_relative','gradient_sha256']},native_cache_environment=p['native_cache_environment'],nominal_operations_per_view=dict(ordinary=ordinary,latent=latent,constant=latent),work_ratio=latent/ordinary,nominal_ordinary_rounding_fraction=updates['ordinary']*ordinary/(800*latent)-1,evaluation_anchors=448,evaluation_groups=evalgroups,evaluation_split='tune',weights=[.6,.2,.2],
 primary='Full32 natural continuation NLL under four fixed independent workspace opportunities vs uniform-NTP ordinary at matched nominal matrix training work.',mechanism='Absolute own-workspace utility on anchor4 vs raw same checkpoint;matched trained constant-query control;all absolute C0 scores reported.',advance_rule='One reused TUNE panel and one trained seed only.Promising mechanism requires positive absolute anchor utility and improvement over trained constant-query and ordinary controls.Broad useful target .01nats;anchor-only .03 with broad deterioration<=.005 is separately labelled mechanism endpoint.No confirmation or automatic scaling on inconclusive results.',
 retirement_policy='After every179-gradient independent mature PASS,retire exact audited64 derivative scratch files.Full final model/AdamW/RNG states and authentic inputs remain.After verified latent800 endpoint and its independent mature PASS,retire only temporary latent101 rolling backup and derivative scratch,preserve all prefix source/RNG metadata and receipt hashes.After constant mature PASS,retire two cold CPU derivative references with exact hashes;authentic inputs/source/audits remain.This is derivative/temporary retirement,not exact archival.Never remove old scientific931 endpoint.',
 prefix_cost_policy='All101 retained latent updates and931 retained ordinary updates,source views and nominal work included once in totals.Original complete assigned path costs reported alongside new continuation;qualification,replay,cold gates,failed work,CPU audits and reservation opportunity separate.No invisible warm-up,no cooled checkpoint extended,no LR reset.',
 scope='Bounded correction of target weighting and cross-anchor dependencies.Uniform159 labels once:32 RNG-selected independent workspace targets+127 ordinary;no PG,teacher,synthetic data,new parameters or learned selection.Latent total800 includes101 qualified retained prefix.Constant800 freshC0/freshAdamW.Ordinary total940 includes931 audited plain-NTP prefix;9 more updates.All full179 mature audits before TUNE scoring.Matched nominal work not walltime;no annealed/tuned convergence,reasoning,allocation or random-init claim.512GiB host,35GiB filesystem,2% named-gradient gates unchanged.')
 out=root/ID;out.mkdir();PUBLIC.mkdir(exist_ok=True)
 for d in [out,PUBLIC]:write_state(d/'plan.json',plan)


def retire(root,arm,plan):
 audit=root/('spliced-workspace-discovery-audit-'+arm+'-v1');proof=json.loads((audit/'report.json').read_text());ap=json.loads((audit/'plan.json').read_text())
 if proof['status']!='PASS_SPLICED_INDEPENDENT_MATURE_CPU_AUDIT':raise AssertionError('Independent179 PASS required before retirement')
 paths=[root/'runs'/plan['arm_runs'][arm]/f'rank{k}-mature-gradients.pt' for k in range(64)];expected=ap['gradient_sha256'];extra=[]
 if arm=='latent':
  prefix=plan['prefixes']['latent'];paths += [root/prefix['run_relative']/f'rank{k}-u101.zip' for k in range(64)];expected += prefix['checkpoint_sha256']
  earlier=json.loads((root/'spliced-workspace-warm-v2-mature-audit/plan.json').read_text());paths += [root/prefix['run_relative']/f'rank{k}-mature-gradients.pt' for k in range(64)];expected += earlier['gradient_sha256']
 if arm=='constant':
  for label in ['spliced-workspace-cpu-reference-v1','spliced-workspace-constant-cpu-reference-v1']:
   r=json.loads((root/label/'report.json').read_text());paths.append(root/r['gradient_relative']);expected.append(r['gradient_sha256'])
 rows=[]
 for path,h in zip(paths,expected,strict=True):
  if path.is_symlink() or file_sha256(path)!=h:raise AssertionError('Only exact registered derivative/temporary paths may retire')
  rows.append(dict(relative=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=h))
 record=dict(status='PREPARED_EXPLICIT_DERIVATIVE_RETIREMENT',arm=arm,files=rows,policy=plan['retirement_policy']);write_state(root/ID/(arm+'-retirement.json'),record)
 subprocess.run(['sudo','-n','rm','--',*[str(f) for f in paths]],check=True)
 record.update(status='COMPLETE_EXPLICIT_DERIVATIVE_RETIREMENT',retired_bytes=sum(v['bytes'] for v in rows))
 for d in [root/ID,PUBLIC]:write_state(d/(arm+'-retirement.json'),record)


def run(root):
 out=root/ID;e=AllChipExecution(root,out/'plan.json',out/'execution');p=e.plan;os.environ.update(p['native_cache_environment']);r=dict(status='STOPPED_SPLICED_WORKSPACE_DISCOVERY',records={},audits={})
 try:
  for arm in ['latent','constant','ordinary']:
   phase=p['arm_runs'][arm];args=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase]),'--arm',arm]
   r['records'][arm]=e.native(phase,'scripts.train_spliced_workspace_discovery',args,'COMPLETE_SPARSE_DISCOVERY_ARM_PENDING_MATURE_HF_AUDIT')
   audit='spliced-workspace-discovery-audit-'+arm+'-v1';r['audits'][arm]=e.cpu(audit,'scripts.audit_spliced_workspace_mature',['--root',str(root),'--plan',str(out/'plan.json'),'--arm',arm],root/audit/'report.json','PASS_SPLICED_INDEPENDENT_MATURE_CPU_AUDIT')
   retire(root,arm,p)
   for d in [out,PUBLIC]:write_state(d/'progress.json',r)
  phase=ID+'-evaluation';r['evaluation']=e.native(phase,'scripts.evaluate_spliced_workspace_discovery',['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase])],'COMPLETE_SPLICED_WORKSPACE_DISCOVERY_EVALUATION')
  from compute_trainium.sparse_workspace_analysis import analyze_sparse
  r['analysis']=analyze_sparse(root,p)
  for d in [out,PUBLIC]:write_state(d/'analysis.json',r['analysis'])
  r['status']='COMPLETE_SPLICED_WORKSPACE_DISCOVERY'
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  r['observed_new_optimizer_updates']={}
  for arm in p['updates']:
   f=root/'runs'/p['arm_runs'][arm]/'updates.json';r['observed_new_optimizer_updates'][arm]=len(json.loads(f.read_text())) if f.exists() else 0
  r['interrupted_update_may_be_unrecorded']=r['status'].startswith('STOPPED');e.close(out/'execution-costs.json');r['plan_sha256']=e.sha
  for d in [out,PUBLIC]:write_state(d/'report.json',r);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
