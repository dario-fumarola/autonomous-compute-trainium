"""Explicit selector mapping for disposable supported-budget qualification."""
import torch

from .budgeted_interleaved_reference import BudgetedInterleavedReference, grouped_layout, matrix_work as reference_work
from .probability_workspace import MathBlock
from .model import ModelConfig

# Keys are compatibility graph identifiers, not iteration counts. Keep both
# scientific quantities explicit in every plan and result.
GRAPH_SPECS = {0: (0, 0), 12: (1, 2), 13: (1, 3), 24: (2, 4), 25: (2, 5)}


def graph_spec(selector, mode):
    if type(selector) is not int or selector not in GRAPH_SPECS or mode != 'latent':
        raise ValueError('Explicit registered generated-state graph identifier required')
    return GRAPH_SPECS[selector]


class BudgetedInterface(BudgetedInterleavedReference):
    def __init__(self, config, *, explicit_attention=False):
        super().__init__(config, explicit_attention=explicit_attention)
        for budget in (1, 2):
            _, allowed = grouped_layout(config.max_seq, budget)
            self.register_buffer(f'budget{budget}_allowed', allowed, persistent=False)
            self.register_buffer(f'budget{budget}_cos', self.cos.repeat(1, 1, budget+1, 1), persistent=False)
            self.register_buffer(f'budget{budget}_sin', self.sin.repeat(1, 1, budget+1, 1), persistent=False)

    def expanded_body(self, source, states):
        if not self.explicit_attention or source.shape[1] != self.config.max_seq:
            return super().expanded_body(source, states)
        budget = len(states)
        if budget not in (1, 2) or any(z.shape != source.shape for z in states):
            raise ValueError('Matching qualified generated slots required')
        h = torch.cat((source, *states), 1)
        cos = getattr(self, f'budget{budget}_cos').to(h.dtype)
        sin = getattr(self, f'budget{budget}_sin').to(h.dtype)
        allowed = getattr(self, f'budget{budget}_allowed')
        for block in self.blocks:
            h = block(h, cos, sin, allowed)
        return MathBlock.rms(h, self.norm).split(source.shape[1], dim=1)

    def forward(self, tokens, selector=0, mode='latent'):
        budget, rounds = graph_spec(selector, mode)
        return super().forward(tokens, budget, rounds)


def make_model(plan, *, explicit_attention=False):
    if plan.get('workspace_mechanism') != 'budgeted_persistent_generated_state':
        raise ValueError('Explicit supported-budget registration required')
    return BudgetedInterface(ModelConfig(**plan['model_config']), explicit_attention=explicit_attention)


def matrix_work(config, length, selector, mode='latent', batch=1):
    budget, rounds = graph_spec(selector, mode)
    return reference_work(config, length, budget, rounds, batch)
