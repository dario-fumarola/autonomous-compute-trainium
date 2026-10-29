"""Explicit plan selection; old plans retain the exact original model class."""
from .model import ModelConfig
from .probability_workspace import ProbabilityWorkspace
from .budget_conditioned_workspace import BudgetConditionedWorkspace
from .untied_probability_workspace import UntiedProbabilityWorkspace

def probability_model(plan,*,explicit_attention=False):
    flag=plan.get('remaining_budget_conditioning',False)
    untied=plan.get('untied_prediction_table',False)
    if type(flag) is not bool or type(untied) is not bool:raise ValueError('Explicit boolean model flags required')
    if flag and untied:raise ValueError('Combining unqualified interventions is not supported')
    c=ModelConfig(**plan['model_config'])
    if untied:return UntiedProbabilityWorkspace(c,explicit_attention=explicit_attention)
    if flag:return BudgetConditionedWorkspace(c,explicit_attention=explicit_attention,maximum_extra_steps=plan.get('maximum_extra_steps',2))
    return ProbabilityWorkspace(c,explicit_attention=explicit_attention)


def gradient_families(plan):
    families=[('embedding',lambda n:n=='embedding.weight'),
              ('body',lambda n:n not in ('embedding.weight','remaining_budget_cues','prediction_weight'))]
    if plan.get('untied_prediction_table',False):
        families.append(('output_table',lambda n:n=='prediction_weight'))
    if plan.get('remaining_budget_conditioning',False):
        families.append(('budget_cues',lambda n:n=='remaining_budget_cues'))
    return families
