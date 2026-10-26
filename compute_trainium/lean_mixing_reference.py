"""Independent pinned-upstream block order for the lean cheap control."""
from types import MethodType
from scripts.audit_parcae_precise_training_point import make_reference as original_reference


def make_reference(config,recurrent):
    if recurrent is not True:raise ValueError('Recurrent-family parameterization required')
    model=original_reference(config,True)
    def iterate(self,input_embeds,freqs,mask,num_steps_pair=None):
        if num_steps_pair is None or tuple(num_steps_pair.shape)!=(2,):raise ValueError('Explicit full-gradient mixing depth required')
        detached,trained=map(int,num_steps_pair)
        if detached!=0 or trained<1:raise ValueError('No detached mixing prefix in this control')
        state=self.initialize_state(input_embeds)
        for _ in range(trained):state=self.transformer.adapter(state,input_embeds)
        offset=self.config.n_layers_in_prelude
        for i,block in enumerate(self.transformer.core_block):
            key=str(offset+i)
            value=self.value_embeds[key](self._current_input_ids) if key in self.value_embeds else None
            state=block(state,freqs,mask,ve=value)
        return state,detached,trained,state.detach()
    model.iterate_forward=MethodType(iterate,model)
    return model
