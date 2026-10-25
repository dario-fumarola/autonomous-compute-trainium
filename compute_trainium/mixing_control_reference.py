"""Independent upstream CPU control: change iteration order,retain its blocks."""
from types import MethodType

from scripts.audit_future_span_qualification import construct_reference as original_reference


def construct_reference(config,state):
    model=original_reference(config,state)
    def iterate(self,input_embeds,freqs,mask,num_steps_pair=None):
        if num_steps_pair is None or tuple(num_steps_pair.shape)!=(2,):
            raise ValueError('Explicit full-gradient mixing depth required')
        detached,trained=map(int,num_steps_pair)
        if detached != 0 or trained < 1:
            raise ValueError('The control never detaches an early mixing step')
        x=self.initialize_state(input_embeds)
        for _ in range(trained):
            x=self.transformer.adapter(x,input_embeds)
        offset=self.config.n_layers_in_prelude
        for i,block in enumerate(self.transformer.core_block):
            key=str(offset+i)
            ve=self.value_embeds[key](self._current_input_ids) if key in self.value_embeds else None
            x=block(x,freqs,mask,ve=ve)
        return x,detached,trained,x.detach()
    model.iterate_forward=MethodType(iterate,model)
    return model
