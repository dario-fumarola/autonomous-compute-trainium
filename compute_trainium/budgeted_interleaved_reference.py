"""CPU-only reference for distinct latent-token budgets and solver rounds.

No trainer, qualification or fit is admitted here. Budget one reproduces the
existing persistent model. Budget zero is the ordinary predictor. A future
same-checkpoint budget comparison requires explicit training support first.
"""
import torch
from torch.nn import functional as F

from .interleaved_workspace import InterleavedWorkspace
from .probability_workspace import MathBlock


def check_budget(budget):
    if type(budget) is not int or budget not in (0, 1, 2):
        raise ValueError('Only the explicit zero/one/two latent-token references')


def grouped_layout(length, budget, device=None):
    check_budget(budget)
    if type(length) is not int or length < 1:
        raise ValueError('Positive source length required')
    positions = torch.arange(length, device=device).repeat(budget+1)
    # Stored by slot type; attend in authentic chronological order.
    order = torch.cat([(budget+1)*torch.arange(length, device=device)+j
                       for j in range(budget+1)])
    return positions, order[None, :] <= order[:, None]


class BudgetedInterleavedReference(InterleavedWorkspace):
    def expanded_body(self, source, states):
        budget = len(states)
        check_budget(budget)
        if source.ndim != 3 or any(z.shape != source.shape for z in states):
            raise ValueError('Matching source and latent shapes required')
        n = source.shape[1]
        if not 0 < n <= self.config.max_seq:
            raise ValueError('Bounded source context required')
        h = torch.cat((source, *states), 1)
        positions, allowed = grouped_layout(n, budget, h.device)
        cos = self.cos.index_select(2, positions).to(h.dtype)
        sin = self.sin.index_select(2, positions).to(h.dtype)
        for block in self.blocks:
            h = block(h, cos, sin, allowed)
        h = MathBlock.rms(h, self.norm) if self.explicit_attention else self.norm(h)
        return h.split(n, dim=1)

    def forward(self, tokens, thought_budget=1, refinements=2, *, return_states=False):
        check_budget(thought_budget)
        if type(refinements) is not int or refinements < 0:
            raise ValueError('Nonnegative numerical refinement count required')
        source = self.embeddings(tokens)
        n = source.shape[1]
        initial = self.body_at(source, torch.arange(n, device=tokens.device))
        if thought_budget == 0:
            logits = F.linear(initial, self.prediction_weight)
            return (logits, ()) if return_states else logits
        states = tuple(initial for _ in range(thought_budget))
        for _ in range(refinements):
            # Output at the source generates latent1; output at latent1
            # generates latent2. This is simultaneous Jacobi substitution.
            states = self.expanded_body(source, states)[:thought_budget]
        final = self.expanded_body(source, states)[thought_budget]
        logits = F.linear(final, self.prediction_weight)
        return (logits, states) if return_states else logits

    def sequential_reference(self, tokens, thought_budget=1, *, return_states=False):
        check_budget(thought_budget)
        source = self.embeddings(tokens)
        history, positions, outputs = [], [], []
        states = [[] for _ in range(thought_budget)]
        for t in range(source.shape[1]):
            history.append(source[:, t:t+1])
            positions.append(t)
            for slot in range(thought_budget):
                z = self.body_at(torch.cat(history, 1),
                                 torch.tensor(positions, device=tokens.device))[:, -1:]
                states[slot].append(z)
                history.append(z)
                positions.append(t)
            final = self.body_at(torch.cat(history, 1),
                                 torch.tensor(positions, device=tokens.device))[:, -1:]
            outputs.append(F.linear(final, self.prediction_weight))
        logits = torch.cat(outputs, 1)
        state_tuple = tuple(torch.cat(s, 1) for s in states)
        return (logits, state_tuple) if return_states else logits


def matrix_work(config, length, thought_budget, refinements, batch=1):
    check_budget(thought_budget)
    if type(refinements) is not int or refinements < 0:
        raise ValueError('Nonnegative numerical refinement count required')
    if type(length) is not int or not 0 < length <= config.max_seq:
        raise ValueError('Bounded positive source length required')
    body = config.flops(0, length, batch)['forward_matmul_flops']
    head = 2*batch*length*config.dim*config.vocab_size
    expanded = (refinements+1)*config.flops(0, (thought_budget+1)*length, batch)['forward_matmul_flops'] if thought_budget else 0
    return dict(forward_matmul_flops=body+expanded+head,
                training_matmul_flops=3*(body+expanded+head),
                thought_budget=thought_budget, numerical_refinements=refinements,
                stored_positions=(thought_budget+1)*length,
                scope='Full dense reference matmuls and backward; no sparse, cached or hardware-speed claim.')
