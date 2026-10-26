# Conventional eight-layer model: CPU parity passes

The explicitly configured eight-layer, width-640, five-head ordinary model
contains 141,733,120 parameters. Its pinned-upstream initialization, predictions,
loss and gradients agree with the local implementation on four authentic TRAIN
windows, both initially and after the same qualification-only gradient
intervention. Largest logit discrepancy is 0.000002861; gradient relative-L2
errors are below 0.000000254. The check took 53.946889 CPU seconds.

The first attempt's configuration failure remains preserved separately. This
version explicitly sets upstream depth before initialization. Probe-updated
weights are not saved or adopted. Sixteen forward and sixteen backward calls,
two manual numerical interventions, zero scientific optimizer updates and zero
native chips. This is implementation parity, not learning or native qualification.
A future independent eight-layer auditor must use the explicit depth constructor.

[Receipt](report.json) · [Protocol](../../../docs/ORDINARY8_CPU_PARITY_20261026.md).
