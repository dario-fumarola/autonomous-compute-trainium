import torch
from compute_trainium.workspace_streaming_checkpoint import save_streaming
from compute_trainium.workspace_cpu_reconstruction import load_streamed_model


def test_full_cpu_reconstruction_has_exact_sharded_and_replicated_coverage(tmp_path):
    torch.manual_seed(7369);model=torch.nn.Linear(8,8);expected={n:p.clone() for n,p in model.named_parameters()}
    paths=[];names=list(expected)
    for rank in range(4):
        path=tmp_path/f'rank{rank}.zip';paths.append(path)
        tensors=[('model.weight',expected['weight'].chunk(4,0)[rank]),('model.bias',expected['bias'])]
        save_streaming(path,tensors,{'parameter_names':names})
    with torch.no_grad():
        for p in model.parameters():p.zero_()
    load_streamed_model(paths,model,label='model')
    for n,p in model.named_parameters():assert torch.equal(p,expected[n])
