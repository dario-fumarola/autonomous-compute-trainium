"""Source-only reserve before joint-exit scores; never opens a model."""
import collections,hashlib,json,resource,sqlite3,time
from pathlib import Path
import numpy as np
from compute_trainium.training import file_sha256
ROOT=Path('/run/user/1000/autonomous-compute/instance-nvme');OUT=Path(__file__).resolve().parent
private=ROOT/'probability-fresh-broad-reserve-v1';private.mkdir(mode=0o700,exist_ok=False)
start=time.time();data=ROOT/'probability-workspace-data-v1';source=ROOT/'natural-append-3250m/extension'
old=json.loads((data/'plan.json').read_text());data_report=json.loads((data/'report.json').read_text())
train_path=data/'PRIVATE-training-documents.jsonl'
assert file_sha256(train_path)==data_report['files'][train_path.name]
train={json.loads(line)['group'] for line in train_path.open()}
paths=[data/'PRIVATE-tune-sources.json',ROOT/'recoverable-dependency-panel-v2/PRIVATE-cases.json',ROOT/'probability-workspace-evaluation-v1/PRIVATE-dependency-metadata.json',ROOT/'probability-structural-panel-v1/PRIVATE-metadata.json']
used={r['group'] for f in paths for r in json.loads(f.read_text())}
protected_path=ROOT/'residual-memory-attribution-v1/PRIVATE-reserved-sources.json';protected={r['group'] for r in json.loads(protected_path.read_text())}
excluded=train|used|protected
plan=dict(status='REGISTERED_SOURCE_ONLY_RESERVE',registered_unix=start,context=512,groups_per_domain=128,
 selection='Accepted source TRAIN metadata with group-first8hex modulo10 zero,excluding current training,all listed previous probability/dependency panels and320protected groups. Hash-rank groups and documents with fixed salt9531;one513-token window per group from first16384 tokens.',
 excluded_membership_sha256={str(f.relative_to(ROOT)):file_sha256(f) for f in [train_path,*paths,protected_path]},
 source_database_sha256=old['accepted_database_sha256'],source_token_sha256=old['raw_token_sha256'],
 tokenizer_revision=old['tokenizer_revision'],script_sha256=file_sha256(__file__),model_scores=0,seconds_limit=600,memory_limit_gib=8,
 scope='Unscored new source groups for a separately registered follow-up only. This is not automatic fit/evaluation admission or a claim that the entire historical project never touched these sources. Do not substitute this reserve for a failed current endpoint. Existing320group confirmation reserve stays sealed.')
for folder in (OUT,private):(folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
assert file_sha256(source/'documents.sqlite')==old['accepted_database_sha256']
for d,h in old['raw_token_sha256'].items():assert file_sha256(source/(d+'.u16'))==h
conn=sqlite3.connect('file:'+str(source/'documents.sqlite')+'?mode=ro',uri=True)
arrays={d:np.memmap(source/(d+'.u16'),dtype='<u2',mode='r') for d in ('web','math','code')}
rank=lambda s:hashlib.sha256((s+':fresh-broad-9531').encode()).hexdigest()
rows=[];views=[];selected=set();eligible_counts={}
for domain in arrays:
 candidates={}
 for identity,group,offset,length in conn.execute('select id,source_group,offset,length from documents where domain=? and split=?',(domain,'train')):
  if length<513 or group in excluded|selected or int(group[:8],16)%10!=0:continue
  key=rank(identity)
  if group not in candidates or key<candidates[group][0]:candidates[group]=(key,identity,offset,length)
 eligible_counts[domain]=len(candidates)
 for group in sorted(candidates,key=rank)[:128]:
  _,identity,offset,length=candidates[group];cap=min(length,16384)
  window=int(rank(identity+':window'),16)%(cap-512)
  view=np.asarray(arrays[domain][offset+window:offset+window+513]).copy()
  assert len(view)==513 and int(view.max())<32000
  rows.append(dict(id=identity,group=group,domain=domain,kind='broad',source_offset=offset,source_window_start=window,target_tokens=512))
  views.append(view);selected.add(group)
 if sum(r['domain']==domain for r in rows)!=128:raise RuntimeError('Source-yield shortfall;reserve remains unscored')
 if time.time()-start>600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise RuntimeError('Preparation bound')
conn.close();assert not selected&excluded and len(selected)==384
np.save(private/'broad-tokens.npy',np.stack(views));(private/'PRIVATE-broad-metadata.json').write_text(json.dumps(rows)+'\n')
# Independently reconstruct every saved window directly from source offsets.
loaded=np.load(private/'broad-tokens.npy',mmap_mode='r')
for i,r in enumerate(json.loads((private/'PRIVATE-broad-metadata.json').read_text())):
 begin=r['source_offset']+r['source_window_start'];assert np.array_equal(loaded[i],arrays[r['domain']][begin:begin+513])
report=dict(status='FROZEN_UNSCORED_BROAD_SOURCE_RESERVE',groups=384,targets=384*512,domains=dict(collections.Counter(r['domain'] for r in rows)),eligible_groups_by_domain=eligible_counts,
 training_overlap=0,previous_listed_panel_overlap=0,protected_overlap=0,model_scores=0,optimizer_updates=0,assigned_chips=0,all_source_windows_reconstructed=True,
 files={f.name:file_sha256(f) for f in private.iterdir() if f.suffix=='.npy' or f.name.startswith('PRIVATE')},wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope=plan['scope'])
for folder in (OUT,private):(folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
