"""Conventional tied-table challenger, with the existing source scaling."""
import math
from .model import ModelConfig
from .probability_workspace import ProbabilityWorkspace
from .interleaved_workspace import InterleavedWorkspace


class TiedOrdinary(InterleavedWorkspace):
    def __init__(self,config,*,explicit_attention=False):
        # Bypass the untied class initializer: there must be no second table,
        # duplicate registered alias or extra optimizer state.
        ProbabilityWorkspace.__init__(self,config,explicit_attention=explicit_attention)
        self.source_scale=math.sqrt(2.5*config.dim)

    @property
    def prediction_weight(self):
        return self.embedding.weight

    def forward(self,tokens,refinements=0,mode='ordinary'):
        if refinements!=0 or mode!='ordinary':
            raise ValueError('Qualified conventional challenger supports ordinary K0 only')
        return self.jacobi(tokens,0,mode='ordinary')


def make_model(plan,*,explicit_attention=False):
    if plan.get('untied_prediction_table') is not False:
        raise ValueError('Tied challenger must explicitly disable an independent output table')
    return TiedOrdinary(ModelConfig(**plan['model_config']),explicit_attention=explicit_attention)
