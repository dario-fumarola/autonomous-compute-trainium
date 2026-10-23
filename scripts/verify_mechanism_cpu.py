"""Independent full-panel FP32 audit of the three new terminal checkpoints."""
import argparse,gc,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from compute_trainium.data import document_windows
from compute_trainium.independent import IndependentModel
from compute_trainium.independent_untied import IndependentUntiedModel
from compute_trainium.metrics import DocumentScore,summarize
from compute_trainium.model import ModelConfig
from compute_trainium.training import file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--models-manifest',type=Path);a=p.parse_args()
 a.output.mkdir(exist_ok=False);torch.set_num_threads(16);torch.set_num_interop_threads(1);started=time.time()
 proof=json.loads((a.root/'runs/independent-untied-cpu-001/report.json').read_text())
 if proof['status']!='PASS_INDEPENDENT_UNTIED_CPU_REFERENCE':raise ValueError('Independent untied reference is not qualified')
 corpus=a.root.parent/'natural-v2-1050m/dev.json';docs=sorted([d for d in json.loads(corpus.read_text()) if d['split']=='dev' and d.get('cohort')=='curve'],key=lambda d:d['id'])
 models=[('untied','untied-depth-v1-s23-u3840'),('ordinary','stable-cont-v2-ordinary-s23-u11520'),('shared','stable-cont-v2-recurrent-s23-u11520')]
 if a.models_manifest:models=json.loads(a.models_manifest.read_text())
 if len(models)!=3 or any(arm not in ['untied','ordinary','shared'] for arm,_ in models):raise ValueError('Require three qualified endpoint paths')
 spec={'models':models,'corpus_sha256':file_sha256(corpus),'started_unix':time.time(),'threads':16,
  'precision':'FP32 weights/matmuls, FP64 log normalization','maximum_native_difference_nats':.0002,
  'source_sha256':{f:file_sha256(f) for f in ['scripts/verify_mechanism_cpu.py','compute_trainium/independent.py','compute_trainium/independent_untied.py','compute_trainium/diagnostics.py']},
  'scope':'Independent numerical verification on the same DEV panel, not new model replications or TEST selection'}
 (a.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n');records=[]
 with torch.inference_mode():
  for arm,run in models:
   phase=a.root/'runs'/(run+'-dev')/'report.json'
   while not phase.exists():
    if time.time()>a.deadline_unix:raise TimeoutError('CPU audit waiting deadline')
    time.sleep(10)
   saved=torch.load(a.root/'runs'/run/'checkpoint.pt',map_location='cpu',mmap=True,weights_only=False);sig=saved['signature']
   if sig['dev_corpus_sha256']!=spec['corpus_sha256']:raise ValueError('DEV source differs')
   config=ModelConfig(**{k:v for k,v in sig['model'].items() if k in ModelConfig.__dataclass_fields__})
   model=(IndependentUntiedModel(config) if arm=='untied' else IndependentModel(config,repetitions=1 if arm=='ordinary' else 2)).eval()
   model.load_titan_state(saved['model']);exposure=saved['real_tokens'];del saved;model_start=time.time();scores=[]
   for i,d in enumerate(docs):
    total=0.;count=0
    for source,mask in document_windows(d['tokens'],1024):
     n=int(mask.sum());hidden=model.hidden(source[None,:n])
     for first in range(0,n,128):
      end=min(n,first+128);logits=F.linear(hidden[:,first:end],model.embedding.weight)[0].double();target=source[first+1:end+1]
      total+=float((logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1)).sum())
     count+=n
    if count!=len(d['tokens']):raise AssertionError('Independent audit target coverage differs')
    scores.append(DocumentScore(d['domain'],d['id'],d['group'],total,count))
    if (i+1)%100==0:print(json.dumps({'run':run,'documents':i+1,'seconds':time.time()-model_start}),flush=True)
    if time.time()>a.deadline_unix:raise TimeoutError('CPU audit scoring deadline')
   metrics=summarize(scores);native=json.loads(phase.read_text());difference=metrics['weighted_nll']-native['metrics']['weighted_nll']
   row={'arm':arm,'run':run,'training_real_targets':exposure,'metrics':metrics,'cpu_minus_native_nll':difference,'seconds':time.time()-model_start,'targets':sum(d.tokens for d in scores)};records.append(row)
   (a.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
   if abs(difference)>.0002:raise AssertionError('Independent full-panel NLL differs from native')
   print(json.dumps({'event':'independent_endpoint','run':run,'cpu_minus_native_nll':difference}),flush=True)
   del model,scores;gc.collect()
 (a.output/'report.json').write_text(json.dumps({'status':'PASS_THREE_NEW_ENDPOINT_CPU_AUDITS','specification':spec,'records':records,'wall_seconds_including_waits':time.time()-started,'active_scoring_seconds':sum(r['seconds'] for r in records)},indent=2)+'\n')


if __name__=='__main__':main()
