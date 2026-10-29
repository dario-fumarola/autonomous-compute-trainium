"""Independently rebuild every saved view from admitted source token offsets."""
import collections,hashlib,json,sqlite3,time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from compute_trainium.training import file_sha256
root=Path('/run/user/1000/autonomous-compute/instance-nvme');panel=root/'probability-structural-panel-v1';source=root/'natural-append-3250m/extension';started=time.time()
report=json.loads((panel/'report.json').read_text());assert all(file_sha256(panel/n)==h for n,h in report['files'].items())
meta=json.loads((panel/'PRIVATE-metadata.json').read_text());xs=np.load(panel/'inputs.npy');ys=np.load(panel/'targets.npy');ms=np.load(panel/'mask.npy');raw=np.memmap(source/'code.u16',mode='r',dtype='<u2');tk=Tokenizer.from_file('/run/user/1000/autonomous-compute/data-v1/tokenizer.json');db=sqlite3.connect('file:'+str(source/'documents.sqlite')+'?mode=ro',uri=True)
cache={};counts=collections.Counter()
for i,r in enumerate(meta):
 if r['parent_id'] not in cache:
  domain,group,offset,length=db.execute('select domain,source_group,offset,length from documents where id=?',(r['parent_id'],)).fetchone();assert domain=='code' and group==r['group']
  ids=np.asarray(raw[offset:offset+min(length,16384)]);text=tk.decode(ids.tolist(),skip_special_tokens=False);encoding=tk.encode(text,add_special_tokens=False)
  assert encoding.ids==ids.tolist();cache[r['parent_id']]=(ids,text,encoding)
 ids,text,encoding=cache[r['parent_id']];assert hashlib.sha256(ids.astype('<u2').tobytes()).hexdigest()==r['source_token_sha256'];assert hashlib.sha256(text.encode()).hexdigest()==r['source_utf8_sha256']
 t=r['target_token'];lo=r['source_prefix_start'];n=r['prefix_tokens'];assert lo==max(0,t-512) and n==t-lo
 assert np.array_equal(xs[i,:n],ids[lo:t]);assert np.array_equal(ys[i,:n],ids[lo+1:t+1]);assert np.all(xs[i,n:]==2) and np.all(ys[i,n:]==2)
 assert ms[i].sum()==1 and ms[i,n-1] and int(ys[i,n-1])==int(ids[t]);a,b=encoding.offsets[t];assert text[a:b].strip()==r['closer_type']
 assert lo<=r['opener_token']<t and r['opener_distance']==t-r['opener_token']
 counts[r['kind']]+=1
assert len(cache)==141 and counts=={'structural_long':141,'structural_short':141}
r=dict(status='PASS_INDEPENDENT_STRUCTURAL_SOURCE_ARRAY_RECONSTRUCTION',source_documents=len(cache),cases=len(meta),kind_counts=dict(counts),original_prefix_and_targets_exact=True,authored_closer_token_verified=True,causal_opener_retained=True,masked_padding_verified=True,model_reads=0,optimizer_updates=0,assigned_chips=0,wall_seconds=time.time()-started,script_sha256=file_sha256(__file__),panel_report_sha256=file_sha256(panel/'report.json'))
Path('results/probability_workspace_20261029/structural-panel-v1/source-reconstruction-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
