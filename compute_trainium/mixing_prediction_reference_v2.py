"""Read-only upstream prediction adapter;depth is prefix+tail in its API."""
from types import MethodType
import torch


def make_reference(config,recurrent):
    if not recurrent:raise ValueError('The full family parameter inventory is required')
    from scripts.audit_parcae_precise_training_point import make_reference as original
    model=original(config,True)
    def iterate(self,source,freqs,mask,num_steps_pair=None):
        if torch.is_grad_enabled():
            raise ValueError('Prediction-only adapter;use the qualified full-gradient training reference')
        if num_steps_pair is None or len(num_steps_pair)!=2:
            raise ValueError('Explicit prediction depth required')
        prefix,tail=map(int,num_steps_pair)
        if prefix<0 or tail<1:raise ValueError('Nonnegative prefix and positive tail required')
        state=self.initialize_state(source)
        for _ in range(prefix+tail):state=self.transformer.adapter(state,source)
        for j,block in enumerate(self.transformer.core_block):
            key=str(self.config.n_layers_in_prelude+j)
            value=self.value_embeds[key](self._current_input_ids) if key in self.value_embeds else None
            state=block(state,freqs,mask,ve=value)
        return state,prefix,tail,state.detach()
    model.iterate_forward=MethodType(iterate,model)
    return model
