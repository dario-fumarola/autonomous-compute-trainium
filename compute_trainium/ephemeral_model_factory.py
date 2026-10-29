"""Explicit adapters for disposable qualification; no existing factory changes."""
from .ephemeral_workspace import EphemeralWorkspace, reused_source_matrix_work
from .model import ModelConfig


class QualifiedEphemeralInterface(EphemeralWorkspace):
    def forward(self, tokens, refinements=0, mode='latent'):
        if type(refinements) is not int or refinements != 0 or mode != 'latent':
            raise ValueError('Ephemeral graph has one generated state, no Jacobi rounds')
        if self.explicit_attention:
            return self.reused_source(tokens)
        # Independent CPU reference recomputes the source history with SDPA.
        return super().forward(tokens)


def make_model(plan, *, explicit_attention=False):
    if plan.get('workspace_mechanism') != 'ephemeral_generated_state':
        raise ValueError('Explicit ephemeral mechanism registration required')
    if plan.get('ephemeral_training_graph') != 'reused_source_kv':
        raise ValueError('Explicit trained graph and cost ledger required')
    return QualifiedEphemeralInterface(ModelConfig(**plan['model_config']),
                                       explicit_attention=explicit_attention)


def matrix_work(config, length, refinements, mode='latent', batch=1):
    if type(refinements) is not int or refinements != 0 or mode != 'latent':
        raise ValueError('Only the ephemeral generated-state graph is registered')
    return reused_source_matrix_work(config, length, batch)
