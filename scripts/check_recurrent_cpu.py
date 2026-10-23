"""CPU checks of the actual TorchTitan wrapper in the isolated native env."""
import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import torch

from compute_trainium.recurrence import RecurrenceConfig, build_recurrent
from scripts.qualify_native import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    c = RecurrenceConfig(dim=32, layers=4, heads=4, kv_heads=2, ffn_dim=64,
                         vocab_size=128, max_seq=64, prelude=1, core=2, coda=1)
    torch.manual_seed(17)
    shared = build_recurrent(c)
    torch.manual_seed(17)
    one = build_recurrent(replace(c, repetitions=1))
    assert digest(shared.state_dict()) == digest(one.state_dict())
    torch.manual_seed(17)
    untied = build_recurrent(replace(c, untied=True))
    x = torch.randint(0, 128, (2, 12))
    z = x.clone(); z[:, 7:] = (z[:, 7:]+3) % 128
    shared.eval()
    a, b, changed = shared(x), untied(x), shared(z)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(a[:, :7], changed[:, :7], atol=1e-6, rtol=1e-5)
    assert (a[:, 7:]-changed[:, 7:]).abs().max() > 1e-4
    a.square().mean().backward(); b.square().mean().backward()
    up = dict(untied.named_parameters())
    for name, param in shared.named_parameters():
        assert param.grad is not None and param.grad.norm() > 0, name
        expected = up[name].grad
        if name.startswith('decoder.layers.'):
            fields = name.split('.'); index = int(fields[2])
            if c.prelude <= index < c.prelude+c.core:
                extra = '.'.join(['extra_layers', str(index-c.prelude), *fields[3:]])
                expected = expected + up[extra].grad
        elif name.startswith(('post_attention.', 'post_ffn.')):
            fields = name.split('.'); index = int(fields[1])
            if c.prelude <= index < c.prelude+c.core:
                extra = '.'.join([fields[0], str(c.layers+index-c.prelude), *fields[2:]])
                expected = expected + up[extra].grad
        torch.testing.assert_close(param.grad, expected, atol=2e-6, rtol=2e-4)
    uninterrupted = copy.deepcopy(shared)
    interrupted = copy.deepcopy(shared)
    oa = torch.optim.AdamW(uninterrupted.parameters(), lr=.001)
    ob = torch.optim.AdamW(interrupted.parameters(), lr=.001)
    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward(); optimizer.step()
    for _ in range(4): update(uninterrupted, oa)
    for _ in range(2): update(interrupted, ob)
    args.output.mkdir(parents=True, exist_ok=True)
    state = {'model': interrupted.state_dict(), 'optimizer': ob.state_dict()}
    torch.save(state, args.output/'tiny-state.pt')
    recovered = build_recurrent(c)
    oc = torch.optim.AdamW(recovered.parameters(), lr=.001)
    state = torch.load(args.output/'tiny-state.pt', weights_only=False)
    recovered.load_state_dict(state['model']); oc.load_state_dict(state['optimizer'])
    for _ in range(2): update(recovered, oc)
    assert digest({'model': uninterrupted.state_dict(), 'optimizer': oa.state_dict()}) == digest({'model': recovered.state_dict(), 'optimizer': oc.state_dict()})
    report = {'status': 'PASS_RECURRENT_CPU', 'causal_future_invariance': True,
              'one_two_pass_initial_weights_identical': True, 'unique_parameter_count': c.parameter_count(),
              'shared_untied_initial_function_and_summed_gradients_match': True,
              'all_parameters_receive_gradients': True, 'full_optimizer_continuation_exact': True}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
