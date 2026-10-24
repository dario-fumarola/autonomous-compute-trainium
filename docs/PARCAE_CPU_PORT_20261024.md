# Parcae architecture port: CPU qualification

This is preparation for a new reproduction, separate from the running
depth-connection-v1 experiment. No current training source or recipe changes.

The static PyTorch candidate implements the pinned release's equal-width
MHA blocks, QK normalization, pairwise RoPE, scaled squared-ReLU, alternating
gated value embeddings, normalized prelude, diagonal injection, C projection,
and tied output head. The recurrent core contains two unique blocks. Forward
depth and backward cap are explicit. The initial random state is supplied,
making numerical replay possible without relying on a device RNG match.

The ordinary reference and recurrent candidate reproduce the published default
inventories:143,141,184 and144,323,136 unique parameters. This is an0.83%
parameter difference; it must remain explicit in comparisons with that ordinary
reference. Depth-one and deeper versions within the recurrent family share the
same144,323,136 parameters. An empirical reproduction still needs a strong
ordinary comparator and applicable inexpensive architectural controls.

## What ran

The full-width CPU check uses768-dimensional states,32,768 vocabulary entries,
and128-position numerical fixtures. Reference weights are perturbed away from
zero output projections, then loaded identically into the candidate. The same
initial state is supplied to both implementations. Ordinary1-pass and recurrent
4-forward/2-backward execution match the pinned upstream reference exactly for
all logits and gradients. Smaller fixtures additionally check1/1,2/2 and4/4.
Changing future tokens leaves earlier predictions identical.

An independent PyTorch operation counter agrees exactly with the matrix-work
ledger for all seven cases, including full vocabulary projection, its backward
pass, gate matrices, repeated B projections, C, attention and forward-only
iterations. Embedding gathers, elementwise work, optimizer, communication and
idle time remain separate. This is not a hardware throughput measurement.

[Qualified exact report](../results/parcae_candidate_cpu_20261024/attempt3-qualified/report.json) ·
[All attempt costs](../results/parcae_candidate_cpu_20261024/execution-costs.json).

The final measured CPU attempt takes22.07s including imports, with3.47GiB peak
child RSS, zero chips and zero optimizer updates. Earlier passing attempts stay
preserved. No generated research corpus, model-quality evaluation or held-out
panel is involved; these are software fixtures.

## Scope and unresolved reproduction choices

Pin the original release at
[69284c1](https://github.com/sandyresearch/parcae/tree/69284c13746e849104f738d6d1a347b1f457df76).
The reference GPT constructor expects its optional Triton-head class. For CPU
checking only, that constructor is replaced in memory with a tied nn.Linear;
the reference's existing PyTorch tied-output forward path stays unchanged.
No upstream files, installed packages, SDK or host driver are modified.

Candidate initialization follows the release's distributions, with a different
RNG draw order. The equivalence claim concerns identical supplied weights and
states, not identical upstream seed initialization. Native kernels, compiled
graph boundaries, sampled per-sequence depth, distributed gradients, optimizer
updates, complete-state recovery and100-warm-update throughput remain unqualified.
Do not launch a scientific fit from this CPU result alone.

The release-code inspection also finds value-embedding gates assigned to Muon;
paper table20 assigns them to AdamW. A future protocol must explicitly choose
which recipe it tests, alongside the previously recorded depth-one sampler and
matmul-accounting differences. This does not diagnose the published results.
[Pinned grouping inspection](../results/parcae_candidate_cpu_20261024/optimizer-group-inspection.json).

The next implementation step is qualification of static recurrent stages and
the specified optimizer on the native backend after the current scientific
campaign releases its chips. Do not replace per-sequence sampling with a new
depth policy without declaring that adaptation. No large new fit, new TEST
access or terminal confirmation attempt is registered here.

Subsequent CPU work has qualified the optimizer and static AOT stages and
prepared independent complete256-window TRAIN gradients. A separate native
qualification is queued after the current scientific fit; it is still not a
learning result. [Updated preparation and cost scope](PARCAE_OPTIMIZER_AND_STAGES_20261024.md).
