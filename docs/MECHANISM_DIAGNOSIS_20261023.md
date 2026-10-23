# Recurrent mechanism diagnosis — new execution window, October 23

Keep the completed three-seed 1B negative result unchanged. This new campaign
runs from about 13:22 UTC with a six-hour ceiling, ending 19:30 UTC. It uses
only this instance and the existing natural corpus. TEST remains sealed.

First complete the already specified 1/2/4-pass DEV inference matrix, for all
three ordinary/recurrent checkpoint pairs. Calibrate distributed readout on
the original paths to within 0.0001 nats. Predictive dependence on two passes
is distinct from superiority to ordinary training. Extrapolation to four
passes is exploratory. No outcome selects a new depth or primary endpoint.

In parallel, inspect saved global pre-clip norms. Logs are sampled, so report
sampled exceedances, not an exact clipping frequency. On frozen natural
prefixes compare both models in identical isolated-document TRAIN/DEV format:
four documents/domain/split, first 512 targets, all three seeds. TRAIN examples
are selected from the beginning of the actual frozen training order and have
been seen by every checkpoint. These small panels are descriptive, not estimates
of full population training NLL. Report residual-state RMS, centered token
variation, between-pass change and cosine. Measure application-specific core
gradients on the first frozen TRAIN prefix/domain at seed23. Copying parameters
for this readout preserves the forward function and splits derivative terms;
cosine/conflict is observational and cannot prove a cause of training loss.

Next qualify the existing untied-core implementation. Tiny actual-wrapper CPU
checks must prove equal initial functions and summed gradient contributions,
causality and nonzero gradients. Native checks require a tiny global-batch
reference, 104 full-model updates including 100 warm updates, replica equality
and fresh-process full optimizer recovery. Preserve failures and costs.

The untied experiment starts from the same seed23 base weights, with the
second core initialized as an exact copy. There are 36 distinct blocks rather
than 20, so it is intentionally NOT parameter matched. It tests sharing plus
the capacity constraint removed by untying; a win does not isolate gradient
conflict. It has exactly the same estimated dense matmul FLOPs as the shared
two-pass path. Use the same source targets/order/global batch, AdamW, clipping,
BF16/FP32 policy and schedule as the completed study. Give it the same three
LR candidates (3e-4, 1e-3, 3e-3), 100M targets each, tuning seed11, and the same
TUNE cohort. Freeze the minimum-NLL LR before a single seed23 1B fit. Require
TUNE NLL below 5.0; otherwise stop before the main fit. The main fit preserves
stable update3456, then anneals for384 updates to3840. Report source-group
uncertainty conditional on one pair, without a three-seed claim. All later
replication is outside this bounded screen.

Complete the predeclared longer diagnostic as ONE seed23 ordinary/recurrent
pair. Resume untouched stable update3456 states, preserving Adam, RNG,
source cursor and past FLOPs. Validate the append-only 3.25B corpus against the
entire original packed prefix. Keep the existing qualified 64-rank plain-AdamW
execution, batch2/accumulation2/context1024. This preserves execution as well
as source history and avoids making a new optimizer a prerequisite. The
measured ordinary/recurrent rates are about834k/604k real targets/sec. Use a
30% training-time margin plus180 seconds per process-isolated train/eval phase.
Require this complete budget and the untied screen to fit before launching.

For the longer pair keep selected LRs 1e-3/3e-4, warmup history and floor0.1.
Train stably through update10368, then anneal over1152 updates to11520. Read
stable DEV at7680 and10368, and the annealed endpoint at11520. Each is paired
at identical actual real-target exposure. A stable intermediate is descriptive;
the terminal 3B comparison is a separate single-pair diagnostic, never a
replacement for the original 1B three-seed primary. No new LR sweep or seed
selection is permitted for this continuation.

Use a detached, finite queue with atomic stage receipts and heartbeats.
Every stage ends in a complete result or an explicit prerequisite failure;
completed stages can be reused on restart. Hardware failures block dependent
fits, not silent recipe changes. CPU analysis can continue independently.
Do not start a stage that cannot finish inside its measured remaining budget.
Resource guards retain512 GiB host available and35 GiB on output filesystems.
Preserve all source snapshots, optimizer states and private recovery copies.
