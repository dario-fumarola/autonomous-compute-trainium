import torch
from compute_trainium.model import ModelConfig
from compute_trainium.probability_workspace import ProbabilityWorkspace
from compute_trainium.pretraining_optimizer import parameter_groups
from compute_trainium.scale_sharded_fixture import cpu_tree
from scripts.qualify_native import digest

def test_native_checkpoint_representation_restores_exactly_on_cpu():
    torch.manual_seed(7);c=ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=23,max_seq=8)
    model=ProbabilityWorkspace(c,explicit_attention=True)
    opt=torch.optim.AdamW(parameter_groups(model,model.embedding.weight),lr=.001,fused=True)
    model(torch.randint(23,(2,8)),1).square().mean().backward();opt.step()
    saved=cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()))
    reader=ProbabilityWorkspace(c);reader.load_state_dict(saved['model'])
    restored_opt=torch.optim.AdamW(parameter_groups(reader,reader.embedding.weight),lr=.001,fused=True)
    restored_opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    restored=cpu_tree(dict(model=reader.state_dict(),optimizer=restored_opt.state_dict(),rng=torch.get_rng_state()))
    assert digest(restored)==digest(saved)
    restored['model']['embedding.weight'][0,0]+=1
    assert digest(restored)!=digest(saved)
