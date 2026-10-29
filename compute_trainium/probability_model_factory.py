"""Explicit plan selection; old plans retain the exact original model class."""
from .model import ModelConfig
from .probability_workspace import ProbabilityWorkspace
from .budget_conditioned_workspace import BudgetConditionedWorkspace

def probability_model(plan,*,explicit_attention=False):
    flag=plan.get('remaining_budget_conditioning',False)
    if type(flag) is not bool:raise ValueError('Explicit boolean conditioning flag required')
    c=ModelConfig(**plan['model_config'])
    if flag:return BudgetConditionedWorkspace(c,explicit_attention=explicit_attention,maximum_extra_steps=plan.get('maximum_extra_steps',2))
    return ProbabilityWorkspace(c,explicit_attention=explicit_attention)


def gradient_families(plan):
    families=[('embedding',lambda n:n=='embedding.weight'),
              ('body',lambda n:n!='embedding.weight' and n!='remaining_budget_cues')]
    if plan.get('remaining_budget_conditioning',False):
        families.append(('budget_cues',lambda n:n=='remaining_budget_cues'))
    return families
