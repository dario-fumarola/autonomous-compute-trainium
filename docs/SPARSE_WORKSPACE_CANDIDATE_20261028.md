# One differentiable latent step at sparse opportunities

Reference: [Pretraining with Token-Level Adaptive Latent Chain-of-Thought,
equations2 and8](https://arxiv.org/html/2602.08220v2). The implementation adapts
its token/latent-step causal mask and common position IDs. This is NOT the full
published method: initially one supported latent step, fixed opportunities,
fixed equal mixing of base/refined states;no router or adaptive penalty.
No published quality result is inherited by this modified OLMo2 candidate.

The ordinary causal prefix produces final states and per-layer KV. Selected
states are continuous inputs to one further pass through the same transformer,
attending to causal prefix KV and allowed earlier/same-position latent keys.
Keys/values remain differentiable. Later original tokens cannot attend latent
states;latent queries cannot attend future original or future-position latent
states. Direct natural-token loss trains the generated state and reader jointly.
Zero workspace is exactly the original predictor;no new parameters or teacher,
notes,vocabulary sampling,policy gradient or Jacobi iteration is introduced.

First prospective schedule: inputs160 real/padded positions,raw159 targets,
four fixed opportunities at127,135,143,151 predicting natural continuation
positions1,9,17,25. These opportunities are known before looking at targets.
Retain ordinary next-token supervision and compare with an ordinary model with
the SAME anchor-target emphasis,plus a trained constant-query control.
A detach-state training ablation can distinguish direct state-generation credit
from reader adaptation via the shared prefix. All costs and parameter inventories
must be measured before a bounded scientific fit is admitted.

CPU gates: unchanged original logits/inventory,mask future-perturbation test,
nonzero direct state credit,checkpointed all-gradient parity,and an independent
DENSE2D-mask recomputation of both forward values and every gradient. Next:
actual-size independent HF oracle,all-rank Native numerical/update/recovery
fixture,100warm updates and mature full-model audit. No scientific fit,quality
claim,from-scratch campaign or learned allocation is admitted by CPU tests.
