# Review draft: Math2 backward differs from its own forward autograd

Local evidence and reproducible draft only. No upstream submission or maintainer
message has been sent. The SDK,drivers and scientific checkpoints are unchanged.

## Minimal CPU reproduction

The standalone script needs only CPU PyTorch and the installed Python source:

```bash
python scripts/repro_sdpa_math2_standalone.py \
  --sdk-source /path/to/torch_neuronx/_sdpa_math.py
```

It imports that file in isolation,without importing the Neuron backend. There
are no repository model imports,corpus contents,checkpoint files or devices.
Seed109 generates valid causal FP16 attention arrays of shape(1,6,32,128),
with identical RMS-normalized queries and keys,and zero dropout. The reference
is autograd through the **identical installed forward**,not another attention
implementation. Actual CPU results are query67.0873%,key61.4149%,value0.0319%
relativeL2 gradient discrepancy. Forward values are identical by construction.
The fixture's small query/key gradient norms make relative errors sensitive;
absolute errors and norms are retained in the full eight-fixture reports.

Inspected source SHA256:
`432468eb0cb4f4323ea91f993c3a377c782cf8bd34eec448130d16bcd17e1b37`.
The installed wheel is torch-neuronx2.14.3.0.282+dc70ffa in Neuron2.33.
The CPU standalone execution uses PyTorch2.9.1;native uses the isolated SDK
PyTorch2.14 environment. The SDK Python formulas are pure PyTorch.

## Isolated CPU cause

The backward correction uses the saved output,which has already been rounded
to the original half dtype. In a diagnostic only,substituting a recomputed
unrounded output into that Python backward reduces query/key errors to0.0743%/
0.0692% on the same32-position fixture. At1024 positions,the corresponding
query error decreases2.2943%→0.0352%. Nothing else in the formula or input changes.

That counterfactual localizes the CPU discrepancy to the saved-output correction.
It is **not** a valid native operator-input substitution or a deployed patch.
It does not establish which implementation change maintainers should choose.
See `scripts/diagnose_sdpa_math2_correction_cpu.py` and the aggregate
`results/sdpa_math2_investigation_20261025/correction-cpu-v1/report.json`.

## Compiled device evidence and limits

With NKI SDPA disabled,actual compiled capture on32 owned logical cores gives
48.4% query-gradient relative error on the same32-position FP16 fixture.
All replica packets agree. A local explicit-FP32-call candidate which passed
CPU still gives37.2% on device and fails the unchanged2% numerical gate.
The difference between that compiled behavior and CPU remains unisolated.

An alternative using explicit FP32 matmul,causal mask,softmax and value matmul,
differentiated through PyTorch autograd,bypasses the SDPA custom backward.
It passes all eight original fixtures on all32 replicas;the32-position
query error is0.0361%. FP32-input controls pass. Full512-window mature model
audits also pass,but aggregate changes are small:one-pass0.454%→0.459%,
recurrence1.763%→1.690%. These operator findings do not explain the completed
language-model ranking or establish any general training-quality improvement.

The complete numerical manifests,source hashes,CPU comparisons,all-replica
aggregates,failed workaround and execution costs are under
`results/sdpa_math2_investigation_20261025/`. Only numerical software fixtures
were generated;they never enter research training or evaluation.
