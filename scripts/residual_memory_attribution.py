"""Frozen memory attribution and unscored source-group reservation."""
import argparse,json,time,os,shutil,resource,sqlite3
from pathlib import Path
from compute_trainium.training import file_sha256
ID='residual-memory-attribution-v1';PUBLIC=Path('results/residual_memory_attribution_20261028')


def guard(root,deadline):
 available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
 if available<512*1024**3 or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3 or os.statvfs(root).f_favail<10000 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>48*1024**3 or time.time()>deadline:raise RuntimeError('Registered CPU memory/storage/time guard')


def reserve_groups(root,out):
 forbidden=set();receipts={}
 for split in ['train','tune','dev']:
  path=root/'workspace-source-views-v2'/(split+'-views.jsonl');receipts[str(path.relative_to(root))]=file_sha256(path)
  for line in path.open():forbidden.add(json.loads(line)['group'])
 for relative in ['workspace-reader-bootstrap-v1/train-views.json','workspace-reader-qualification-v1/views.json']:
  path=root/relative;receipts[relative]=file_sha256(path);forbidden.update(v['group'] for v in json.loads(path.read_text()))
 path=root/'workspace-policy-gradient-reference-v1/private-source-view.json';receipts[str(path.relative_to(root))]=file_sha256(path);forbidden.add(json.loads(path.read_text())['group'])
 path=root/'recoverable-dependency-panel-v2/PRIVATE-cases.json';forbidden.update(c['group'] for c in json.loads(path.read_text()))
 quotas=dict(web=192,math=64,code=64);selected=[];counts=dict.fromkeys(quotas,0);seen=set();databases={}
 expected=json.loads((root/'workspace-source-views-v2/plan.json').read_text())['accepted_database_sha256']
 for folder in ['natural-append-3250m','natural-append-6500m-v2','natural-append-17000m-v3','natural-append-capacity-v4']:
  path=root/folder/'extension/documents.sqlite'
  if not path.exists():path=root/folder/'documents.sqlite'
  sha=file_sha256(path)
  if sha!=expected[folder]:raise AssertionError('Accepted document metadata changed')
  databases[str(path.relative_to(root))]=sha;c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
  try:
   for domain,quota in quotas.items():
    if counts[domain]>=quota:continue
    for identity,group,length in c.execute("select id,source_group,length from documents where domain=? and split='train' and length>=512 order by id",(domain,)):
     if group in forbidden or group in seen or int(group[:8],16)%10!=0:continue
     selected.append(dict(id=identity,group=group,domain=domain,accepted_database=str(path.relative_to(root)),old_tokenizer_length=length));seen.add(group);counts[domain]+=1
     if counts[domain]==quota:break
  finally:c.close()
  if counts==quotas:break
 if counts!=quotas:raise AssertionError('Insufficient unscored source reserve;no silent refill')
 path=out/'PRIVATE-reserved-sources.json';path.write_text(json.dumps(selected,indent=2)+'\n')
 return dict(status='RESERVED_UNSCORED_SOURCE_IDENTIFIERS',groups=len(seen),by_domain=counts,existing_workspace_group_overlap=0,excluded_groups=len(forbidden),private_manifest_sha256=file_sha256(path),accepted_database_sha256=databases,excluded_source_receipts=receipts,model_scores_seen=False,tokenized_panel_ready=False,scope='Unused by all existing workspace views/calibration/dependency panels;not a claim of unseen upstream foundation or old small-model training data.TEST and old confirmation reserve not opened.')


def prepare(root):
 from compute_trainium.execution_isolation import managed_running
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.residual_memory_intervention import choose_donors
 if managed_running() or os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('Closed owner,CPU-only diagnostic required')
 started=time.time();guard(root,started+1800);out=root/ID;out.mkdir(mode=0o700);PUBLIC.mkdir()
 parent=json.loads((root/'residual-data-volume-v1/plan.json').read_text());data=WorkspaceViewBatches(root/parent['data_relative'],'tune');broad=[dict(stratum=data.at(i)['domain'],group=g,length=159) for i,g in enumerate(parent['evaluation_groups'])];bd,be=choose_donors(broad)
 if be:raise AssertionError('Every original TUNE view needs two independent donors')
 panel=root/'recoverable-dependency-panel-v2/PRIVATE-cases.json';cases=json.loads(panel.read_text());dd,de=choose_donors([dict(stratum=c['kind'],group=c['group'],length=len(c['full_prefix'])+len(c['answer'])-1) for c in cases]);coverage={}
 for kind in sorted({c['kind'] for c in cases}):
  coverage[kind]={str(cohort):dict(cases=sum(str(i) in dd and c['kind']==kind and c['cohort']==cohort for i,c in enumerate(cases)),source_groups=len({c['group'] for i,c in enumerate(cases) if str(i) in dd and c['kind']==kind and c['cohort']==cohort})) for cohort in [0,1]}
 paths=dict(c0=parent['checkpoint_relative'],one='residual-data-volume-v1/PRIVATE-fresh_one-state.pt',local='residual-local-control-v1/PRIVATE-local-state.pt',prior_one='residual-data-volume-v1/PRIVATE-fresh_one-scores.json',prior_local='residual-local-control-v1/PRIVATE-local-scores.json',dependency_cases=str(panel.relative_to(root)),prior_dependency='recoverable-dependency-cpu-v1/c0-metrics.json')
 bindings={name:dict(relative=relative,sha256=file_sha256(root/relative)) for name,relative in paths.items()}
 if bindings['c0']['sha256']!=parent['checkpoint_sha256']:raise AssertionError('Original C0 changed')
 feature_phase=parent['feature_segments']['tune'][0];features={f'runs/{feature_phase}/PRIVATE-rank{k}-tune.pt':file_sha256(root/'runs'/feature_phase/f'PRIVATE-rank{k}-tune.pt') for k in range(64)}
 reserve=reserve_groups(root,out);sources={str(f):file_sha256(f) for folder in ['compute_trainium','scripts','configs'] for f in Path(folder).rglob('*.py')}
 p=dict(identity=ID,registered_unix=started,absolute_deadline_unix=started+1800,scoring_seconds_limit=1200,optimizer_updates=0,assigned_chips=0,bindings=bindings,feature_sha256=features,feature_phase=feature_phase,model_config=parent['model_config'],evaluation_groups=parent['evaluation_groups'],broad_donors=bd,dependency_donors=dd,dependency_excluded_indices=de,dependency_coverage=coverage,width=128,source_sha256=sources,policy_sha256=file_sha256('docs/RESIDUAL_MEMORY_ATTRIBUTION_20261028.md'),scope=Path('docs/RESIDUAL_MEMORY_ATTRIBUTION_20261028.md').read_text(),reserve=reserve)
 for folder in [out,PUBLIC]:
  (folder/'plan.json').write_text(json.dumps(p,indent=2)+'\n');(folder/'source-reserve.json').write_text(json.dumps(reserve,indent=2)+'\n');(folder/'admission.json').write_text(json.dumps(dict(status='PASS_FROZEN_MEMORY_DIAGNOSTIC_ADMISSION',wall_seconds=time.time()-started,dependency_cases=len(dd),excluded=len(de),optimizer_updates=0,assigned_chips=0),indent=2)+'\n')
 print(json.dumps(dict(dependency_coverage=coverage,dependency_excluded=de,reserved_groups=reserve['groups'])))


def summarize(rows,*,broad=False):
 from compute_trainium.residual_screen_analysis import contrast
 from compute_trainium.dependency_analysis import interval
 fields=['raw','local','authentic','donor1','donor2','self_only'];result={}
 for suffix in ['nll','first_nll']:
  sample=rows if suffix=='nll' else [dict(r,real_targets=1) for r in rows]
  reduce=lambda values:contrast(sample,values) if broad else interval(sample,values)
  scores={f:reduce([r[f+'_'+suffix] for r in rows]) for f in fields};contrasts={}
  for name,left,right in [('authentic_utility','raw','authentic'),('local_utility','raw','local'),('local_minus_authentic','local','authentic'),('donor1_minus_authentic','donor1','authentic'),('donor2_minus_authentic','donor2','authentic'),('self_only_minus_authentic','self_only','authentic'),('donor1_utility','raw','donor1'),('donor2_utility','raw','donor2'),('self_only_utility','raw','self_only')]:contrasts[name]=reduce([r[left+'_'+suffix]-r[right+'_'+suffix] for r in rows])
  result[suffix]=dict(scores=scores,contrasts=contrasts)
 return result


def run(root):
 import torch
 from compute_trainium.residual_workspace import ResidualWorkspace
 from compute_trainium.residual_local_control import LocalResidualCorrection
 from compute_trainium.residual_memory_intervention import read_with_memory
 from compute_trainium.stable_target_loss import stable_target_log_probs
 out=root/ID;p=json.loads((out/'plan.json').read_text());start=time.time();deadline=min(start+p['scoring_seconds_limit'],p['absolute_deadline_unix']);torch.set_num_threads(16);report=dict(status='STOPPED_MEMORY_ATTRIBUTION',optimizer_updates=0,assigned_chips=0);versions=[]
 def progress(stage,index):guard(root,deadline);(out/'progress.json').write_text(json.dumps(dict(stage=stage,index=index,wall_seconds=time.time()-start)))
 try:
  for name,sha in p['source_sha256'].items():
   if file_sha256(name)!=sha:raise AssertionError('Frozen diagnostic source changed')
  for v in p['bindings'].values():
   if file_sha256(root/v['relative'])!=v['sha256']:raise AssertionError('Frozen scientific artifact changed')
  for name,sha in p['feature_sha256'].items():
   if file_sha256(root/name)!=sha:raise AssertionError('Frozen feature cache changed')
  one=ResidualWorkspace(mode='one').eval();local=LocalResidualCorrection().eval()
  for name,m in [('one',one),('local',local)]:m.load_state_dict(torch.load(root/p['bindings'][name]['relative'],weights_only=True)['model']);m.requires_grad_(False);versions.append((m,tuple(v._version for v in m.parameters())))
  base_state=torch.load(root/p['bindings']['c0']['relative'],weights_only=True,mmap=True)['model'];head=base_state['lm_head.weight'].detach();head_version=head._version
  h=torch.empty(448,159,2048);y=torch.empty(448,159,dtype=torch.long);domains=torch.empty(448,dtype=torch.long);seen=[]
  for name in p['feature_sha256']:
   v=torch.load(root/name,weights_only=True,mmap=True);ix=v['indices'];h[ix]=v['features'];y[ix]=v['targets'];domains[ix]=v['domains'];seen+=ix.tolist()
  if sorted(seen)!=list(range(448)):raise AssertionError('Complete TUNE feature coverage required')
  parity=dict(max_branch_output_error=0.,max_broad_saved_nll_error=0.,max_dependency_raw_nll_error=0.);prior_one=json.loads((root/p['bindings']['prior_one']['relative']).read_text());prior_local=json.loads((root/p['bindings']['prior_local']['relative']).read_text())
  def score(features,anchors,targets,donors):
   authentic=read_with_memory(one,features,anchors);err=float((authentic-one(features,anchors)).abs().max());parity['max_branch_output_error']=max(parity['max_branch_output_error'],err)
   if err>1e-5:raise AssertionError('Authentic branch identity failed')
   states=dict(raw=features[:,anchors],local=local(features,anchors),authentic=authentic,donor1=read_with_memory(one,features,anchors,donor=donors[0]),donor2=read_with_memory(one,features,anchors,donor=donors[1]),self_only=read_with_memory(one,features,anchors,self_only=True));values={}
   for name,state in states.items():
    nll=-stable_target_log_probs(state@head.T,targets);values[name+'_nll']=float(nll.mean());values[name+'_first_nll']=float(nll[0,0])
   if not all(__import__('math').isfinite(x) for x in values.values()):raise FloatingPointError('Nonfinite memory diagnostic')
   return values
  with torch.no_grad():
   broad=[];anchors=torch.arange(127,159);tick=time.time()
   for i in range(448):
    progress('broad',i);di=p['broad_donors'][str(i)];v=score(h[i:i+1],anchors,y[i:i+1,127:159],[h[j:j+1] for j in di]);err=max(abs(v[k]-old[field]) for old,mapping in [(prior_one[i],{'raw_nll':'raw_nll','authentic_nll':'workspace_nll','raw_first_nll':'first_raw_nll','authentic_first_nll':'first_workspace_nll'}),(prior_local[i],{'local_nll':'workspace_nll','local_first_nll':'first_workspace_nll'})] for k,field in mapping.items());parity['max_broad_saved_nll_error']=max(parity['max_broad_saved_nll_error'],err)
    if err>1e-5:raise AssertionError('Original saved broad predictions changed')
    broad.append(dict(index=i,domain=int(domains[i]),group=p['evaluation_groups'][i],real_targets=32,**v))
    if i==15 and time.time()+(time.time()-tick)/16*432+300>deadline:raise RuntimeError('Measured broad panel does not fit fixed cap')
   (out/'PRIVATE-broad-scores.json').write_text(json.dumps(broad));report['broad']=summarize(broad,broad=True);report['domains']={name:summarize([r for r in broad if r['domain']==i]) for i,name in enumerate(['web','math','code'])};report['broad_seconds']=time.time()-tick;del h,y
   from transformers import AutoModelForCausalLM
   progress('restore_HF_C0',0);backbone=AutoModelForCausalLM.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval();backbone.load_state_dict(base_state,assign=True);backbone.requires_grad_(False);versions.append((backbone,tuple(v._version for v in backbone.parameters())));del base_state
   cases=json.loads((root/p['bindings']['dependency_cases']['relative']).read_text());prior={r['index']:r for r in json.loads((root/p['bindings']['prior_dependency']['relative']).read_text()) if r['condition']=='full' and r['variant']=='original'};features={};tick=time.time()
   for i,c in enumerate(cases):
    progress('dependency_features',i);tokens=torch.tensor(c['full_prefix']+c['answer'][:-1]).reshape(1,-1);features[i]=backbone.model(tokens,use_cache=False).last_hidden_state;anchors=torch.arange(len(c['answer']))+len(c['full_prefix'])-1;targets=torch.tensor(c['answer']).reshape(1,-1);nll=-stable_target_log_probs(features[i][:,anchors]@head.T,targets);err=max(abs(float(nll.mean())-prior[i]['raw_nll']),abs(float(nll[0,0])-prior[i]['raw_first_nll']));parity['max_dependency_raw_nll_error']=max(parity['max_dependency_raw_nll_error'],err)
    if err>.0002:raise AssertionError('Independent historical CPU C0 prediction failed')
    if i==7 and time.time()+(time.time()-tick)/8*(len(cases)-8)+120>deadline:raise RuntimeError('Measured dependency feature budget fails')
   report['dependency_feature_seconds']=time.time()-tick;dep=[];tick=time.time()
   for i,c in enumerate(cases):
    if str(i) not in p['dependency_donors']:continue
    progress('dependency_scoring',i);length=features[i].shape[1];anchors=torch.arange(len(c['answer']))+len(c['full_prefix'])-1;targets=torch.tensor(c['answer']).reshape(1,-1);v=score(features[i],anchors,targets,[features[j][:,:length] for j in p['dependency_donors'][str(i)]]);dep.append(dict(index=i,group=c['group'],kind=c['kind'],cohort=c['cohort'],real_targets=len(c['answer']),**v))
   (out/'PRIVATE-dependency-scores.json').write_text(json.dumps(dep));report['dependency_scoring_seconds']=time.time()-tick;report['dependencies']={};decisions={}
   for kind in sorted({r['kind'] for r in dep}):
    report['dependencies'][kind]={str(cohort):summarize([r for r in dep if r['kind']==kind and r['cohort']==cohort]) for cohort in [0,1]}
    checks=[]
    for cohort in [0,1]:
     v=report['dependencies'][kind][str(cohort)]['nll']['contrasts'];u=v['authentic_utility'];cover=p['dependency_coverage'][kind][str(cohort)];checks.append(cover['cases']>=32 and cover['source_groups']>=12 and u['mean']>=.03 and u['conditional_source_group_95_interval'][0]>0 and all(v[d+'_minus_authentic']['conditional_source_group_95_interval'][0]>0 for d in ['donor1','donor2']))
    decisions[kind]=dict(replicated_mechanism_slice_gate=all(checks),primary=kind in ['code_binding','math_conclusion'])
  if head._version!=head_version or any(tuple(v._version for v in m.parameters())!=version for m,version in versions):raise AssertionError('Inference changed trained parameters')
  for name in ['c0','one','local']:
   v=p['bindings'][name]
   if file_sha256(root/v['relative'])!=v['sha256']:raise AssertionError('Diagnostic altered a saved scientific checkpoint')
  report.update(status='COMPLETE_FROZEN_MEMORY_ATTRIBUTION',parity=parity,parameters_unchanged=True,decisions=decisions,source_reserve_scored=False,confirmation=False,iteration_claim=False,scope='Fixed weights/reused development sources.Content replacement is an intervention with distribution-shift limitations.No fitted controller,extra training or efficiency claim.')
 except BaseException as error:report.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  report.update(wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,plan_sha256=file_sha256(out/'plan.json'))
  for folder in [out,PUBLIC]:(folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
