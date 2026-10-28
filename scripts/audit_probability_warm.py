"""Independent complete-global mature gradient audit, after Native exits."""
import argparse,json,resource,time
from pathlib import Path


def main(a):
    import torch
    from compute_trainium.model import ModelConfig
    from compute_trainium.probability_workspace import ProbabilityWorkspace
    from compute_trainium.probability_workspace_data import ProbabilitySources
    from scripts.qualify_trained_gradient import compare_gradients
    from compute_trainium.training import file_sha256
    start=time.time();torch.set_num_threads(32);p=json.loads(a.plan.read_text())
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen audit plan differs')
    a.output.mkdir(exist_ok=False);data=ProbabilitySources(a.root/p['data_relative'])
    parent=a.root/'runs'/p['warm_phase'];saved=torch.load(parent/'PRIVATE-u103.pt',map_location='cpu',weights_only=False,mmap=True)
    model=ProbabilityWorkspace(ModelConfig(**p['model_config']));model.load_state_dict(saved['model']);del saved
    native=torch.load(parent/'PRIVATE-mature-gradient.pt',weights_only=True,mmap=True)
    count=data.target_count(103*64*p['batch'],64*p['batch']);total=0.
    for rank in range(64):
        indices=list(range((103*64+rank)*p['batch'],(103*64+rank+1)*p['batch']))
        x,y,mask=data.batch(indices);logits=model(x,2)
        value=(torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)*mask).sum()/count
        value.backward();total+=float(value.detach());del logits,value
        if rank%8==0:(a.output/'progress.json').write_text(json.dumps(dict(rank=rank,seconds=time.time()-start)))
        if time.time()-start>p['cpu_phase_seconds_limit'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>48*1024**3:
            raise RuntimeError('Global CPU audit bound')
    expected={n:v.grad for n,v in model.named_parameters()};actual=native['gradients']
    comparisons={'overall':compare_gradients(actual,expected)}
    for family,match in [('embedding',lambda n:n=='embedding.weight'),('body',lambda n:n!='embedding.weight')]:
        comparisons[family]=compare_gradients({n:v for n,v in actual.items() if match(n)},{n:v for n,v in expected.items() if match(n)})
    ok=max(v['relative_l2'] for v in comparisons.values())<=.02 and abs(total-native['loss'])<=.003
    result=dict(status='PASS_PROBABILITY_MATURE_GLOBAL_CPU' if ok else 'FAIL_PROBABILITY_MATURE_GLOBAL_CPU',comparisons=comparisons,
        native_loss=native['loss'],cpu_loss=total,loss_error=total-native['loss'],real_targets=count,
        global_windows=64*p['batch'],assigned_chips=0,optimizer_updates=0,wall_seconds=time.time()-start)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not ok:raise AssertionError('Mature whole-global gradient gate failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
