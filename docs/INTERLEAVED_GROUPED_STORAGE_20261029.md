# Equivalent grouped storage, prospective actual-shape qualification

The controlled body fixture localizes a Native execution cliff to interleaving
before the body: roughly1.374 seconds versus0.011 for the same six-block body
without that preceding layout operation. Grouped history with the matching
permutation of rotary positions and causal access runs near0.0097 seconds in
the fixture and passes CPU gradients. This does not establish full-model speed:
the fixture's split/rejoin may be simplified by the compiler.

The candidate stores `[E_0,...,E_n,z_0,...,z_n]`, with chronological ranks
`[0,2,...,2n,1,3,...,2n+1]`. A query may read a key iff the key's chronological
rank is no greater. Rotary positions remain the original source position for
both slots. Outputs and refined states are returned in original source order.
There are no extra parameters, detached states, changed targets, reduced
precision or omitted matrix operations. The original chronological layout
remains the independent CPU SDPA/CE and sequential reference.

CPU tests cover supported modes/refinement counts, all gradients, three AdamW
steps, future invariance and the converged sequential limit. Adopt only after
cold/mature all-family gradient checks,100 actual-shape warm updates, exact
separate-process full-state replay, all-replica equality and no unexpected
fallback. The0.8-second early throughput screen remains unchanged. Test the
constant-pause mode separately before any learning comparison.

This is a software-layout correction, not another scientific architecture.
Its registered qualification states are disposable. No held-out score or
scientific training is admitted here; original failed attempts and costs remain.
