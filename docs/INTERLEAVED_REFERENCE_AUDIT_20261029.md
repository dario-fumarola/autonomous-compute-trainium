# Pinned reference comparison

Rechecked the [pinned PonderLM-2 implementation](https://github.com/LUMIA-Group/PonderLM-2/blob/fa784bece621b989fb008c59b0fd8d282fa9c73c/src/llamafactory/model/modeling/modeling_gpt_neox.py)
while the registered tuning queue remained unchanged.

Its hidden-state route scales source embeddings by sqrt(2.5*width), as ours
does. Its random refinement option samples2,3,4; the paper's Appendix E uses
2,3. Detached copies in that loop support diagnostic bookkeeping; the refined
state assigned into the next pass remains differentiable. Our full-gradient
implementation does not add a stop-gradient absent from that route.

Initialization differs deliberately: the reference uses truncated normal with
standard deviation sqrt(0.4/width), and additional output-projection scaling by
layer and latent counts. Ours retains the repository's normal0.02 initializer
and output scaling by sqrt(2*layers), identically for candidate and controls.
At width512, the initial source-embedding RMS is therefore about0.715 here,
versus roughly1 for the scaled reference initializer. Architecture, tokenizer,
data and normalization also differ. This is not a reproduction.

Neither the reference audit nor the exploratory parameter-scale observations
prove why a learned endpoint converges slowly. Preserve the single-change
mixed-round comparison before considering an initializer/scale intervention.
That alternative would require its own ordinary and pause controls.
