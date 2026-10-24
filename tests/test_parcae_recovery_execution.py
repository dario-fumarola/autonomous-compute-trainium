import torch

from scripts.run_parcae_optimizer_recovery import replay_comparison


def test_recovery_comparison_charges_update_units_and_each_family(tmp_path):
    cpu=tmp_path/'cpu';primary=tmp_path/'primary';replay=tmp_path/'replay'
    for p in [cpu/'ordinary',primary,replay]:p.mkdir(parents=True)
    def state(step):
        return {'model':{'embed':torch.ones(2)*100-step*.1,'matrix':torch.ones(2)-step*.001},
            'moments':{'embed':{'step':step,'m':torch.ones(2),'v':torch.ones(2)},
                       'matrix':{'m':torch.ones(2),'v':torch.ones(2)}}}
    for step in [2,3]:
        s=state(step);torch.save(s,cpu/'ordinary'/f'cpu-step-{step}.pt');torch.save(s,primary/f'native-step-{step}.pt')
    s=state(3);torch.save(s,replay/'native-step-3.pt')
    assert replay_comparison(cpu,primary,replay,'ordinary')['passes_registered_gates']
    s['model']['matrix']+=.0002;torch.save(s,replay/'native-step-3.pt')
    result=replay_comparison(cpu,primary,replay,'ordinary')
    assert not result['passes_registered_gates']
    assert result['replay_vs_cpu']['families']['muon']['update_comparison']['relative_l2']>.19
