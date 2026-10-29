"""Independent full-mask CPU and cached Native paths for qualification."""
from .depth_causal_workspace import DepthCausalWorkspace, matrix_work as cached_work
from .budgeted_interleaved_reference import check_budget
from .model import ModelConfig


class DepthCausalInterface(DepthCausalWorkspace):
    def forward(self, tokens, budget=0, mode='latent'):
        check_budget(budget)
        if mode != 'latent':
            raise ValueError('Explicit depth-causal graph required')
        if self.explicit_attention:
            return super().forward(tokens, budget)
        return self.full_reference(tokens, budget)


def make_model(plan, *, explicit_attention=False):
    if plan.get('workspace_mechanism') != 'depth_causal_generated_state':
        raise ValueError('Explicit depth-causal registration required')
    return DepthCausalInterface(ModelConfig(**plan['model_config']), explicit_attention=explicit_attention)


def matrix_work(config, length, budget, mode='latent', batch=1):
    if mode != 'latent':
        raise ValueError('Explicit depth-causal ledger required')
    return cached_work(config, length, budget, batch)
