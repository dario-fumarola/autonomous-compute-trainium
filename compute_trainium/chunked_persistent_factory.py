"""Explicit qualification graphs; graph IDs are not thought budgets."""
from .chunked_persistent_workspace import ChunkedPersistentWorkspace, matrix_work as work
from .model import ModelConfig
from .tied_interleaved_ordinary import TiedOrdinary

STRIDE = 128
GRAPH_NAMES = {0:'full_ordinary',1:'chunked_ordinary',2:'generated_slot',3:'constant_slot'}


def validate_graph(graph,mode):
    if type(graph) is not int or graph not in GRAPH_NAMES or mode != 'registered':
        raise ValueError('Explicit registered graph ID required')


class ChunkedInterface(ChunkedPersistentWorkspace):
    def forward(self,tokens,graph=0,mode='registered'):
        validate_graph(graph,mode)
        if graph == 0:
            return TiedOrdinary.forward(self,tokens)
        steps = int(graph in (2,3))
        proposal = 'pause' if graph == 3 else 'latent'
        if self.explicit_attention:
            return self.cached(tokens,STRIDE,steps,proposal)
        return self.full_reference(tokens,STRIDE,steps,proposal)


def make_model(plan,*,explicit_attention=False):
    if (plan.get('workspace_mechanism') != 'chunked_persistent_generated_state'
            or plan.get('untied_prediction_table') is not False or plan.get('source_stride') != STRIDE):
        raise ValueError('Explicit tied-table sparse persistent registration required')
    return ChunkedInterface(ModelConfig(**plan['model_config']),explicit_attention=explicit_attention)


def matrix_work(config,length,graph,mode='registered',batch=1):
    validate_graph(graph,mode)
    return work(config,length,STRIDE,int(graph in (2,3)),batch,ordinary_full=graph==0)
