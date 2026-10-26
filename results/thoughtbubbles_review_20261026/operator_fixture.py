"""Four tiny numerical operator calls;not training or research examples."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

started=time.time()
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
base=Path(__file__).parent
registration=json.loads((base/'registration.json').read_text())
source=base/'upstream-model.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==registration['upstream_source_sha256']
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==registration['fixture_sha256']
assert not (base/'report.json').exists()
tree=ast.parse(source.read_text())
block=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ForkingBlock')
methods=[]
for name in ['clipped_logsigmoid','fork']:
    method=copy.deepcopy(next(n for n in block.body if isinstance(n,ast.FunctionDef) and n.name==name))
    method.decorator_list=[]
    methods.append(method)
namespace=dict(torch=torch,F=F)
exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(source),'exec'),namespace)
self=SimpleNamespace(config=SimpleNamespace(detach_scores=False,gather_by_orig=False,fork_map='static',max_thought_budget=2,block_size=2),
    forking_ln=torch.nn.LayerNorm(2),forking_score=torch.nn.Linear(2,2,bias=False),fork_embedding=torch.tensor([.1,.2]),
    training=False,clipped_logsigmoid=namespace['clipped_logsigmoid'])
with torch.no_grad():self.forking_score.weight.copy_(torch.tensor([[10.,0.],[10.,0.]]))
low=torch.tensor([[[0.,0.],[0.,0.],[-1.,1.],[-1.,1.]]])
high=low.clone();high[:,2:]=torch.tensor([1.,-1.])
scores=torch.full((1,4),-.5)
indices=torch.tensor([[0,0,1,1]])
calls=[]
with torch.no_grad():
    for data in [low,high]:
        calls.append(namespace['fork'](self,data,scores,indices))
    for data in [low,high]:
        calls.append(namespace['fork'](self,data[:,:2],scores[:,:2],indices[:,:2],force_thought_budget=2))
assert torch.equal(low[:,:2],high[:,:2])
counts=[int((v[2]==0).sum()) for v in calls[:2]]
assert counts==[3,1],counts
assert all(torch.equal(a,b) for a,b in zip(calls[2],calls[3]))
assert time.time()-started<registration['seconds_limit']
report=dict(status='PASS_EXACT_RELEASE_FORK_OPERATOR_SUFFIX_DEPENDENCE_FIXTURE',
    upstream_commit=registration['upstream_commit'],upstream_source_sha256=registration['upstream_source_sha256'],
    fixture_sha256=registration['fixture_sha256'],unchanged_prefix_inputs=True,
    full_block_prefix_stream_counts=counts,prefix_only_outputs_identical=True,fork_operator_calls=4,
    model_forwards=0,optimizer_updates=0,research_examples=0,assigned_chips=0,wall_seconds=time.time()-started,
    cpu_seconds=time.process_time(),peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
    scope='The unchanged full-block release fork method can allocate different streams to an earlier source position when only later position inputs change. Direct AST extraction removes method decorators only;no function body is edited. This is a numerical software fixture,not an end-to-end model/evaluation reproduction,a quality result or evidence against recomputed-prefix inference. No Native port or training is admitted.')
(base/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
