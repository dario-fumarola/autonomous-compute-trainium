"""Prefix-only greedy answer bookkeeping;no gold-dependent stopping."""
from dataclasses import dataclass,field
from .natural_chain_views import normalized_answer

@dataclass
class GreedyConclusion:
    prefix:tuple
    context:int=128
    maximum_new_tokens:int=24
    eos:int=2
    generated:list=field(default_factory=list)
    ended:bool=False

    def __post_init__(self):
        if not self.prefix or len(self.prefix)>self.context or self.maximum_new_tokens<1:
            raise ValueError('Valid visible prompt and fixed generation budget required')

    @property
    def limit(self):return min(self.maximum_new_tokens,self.context-len(self.prefix)+1)
    @property
    def done(self):return self.ended or len(self.generated)>=self.limit
    @property
    def prediction_position(self):return len(self.prefix)+len(self.generated)-1

    def input_ids(self):
        if self.done:raise ValueError('Generation already terminated')
        values=[*self.prefix,*self.generated]
        return values+[self.eos]*(self.context-len(values))

    def advance(self,token):
        if self.done:raise ValueError('Cannot consume a token after stopping')
        if token==self.eos:self.ended=True
        else:self.generated.append(int(token))


def correct_scalar(generated_text,original_answer,ended):
    if not ended:return False
    try:return normalized_answer(generated_text)==normalized_answer(original_answer)
    except ValueError:return False
