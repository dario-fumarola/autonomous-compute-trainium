# Read-only inference-cost qualification

Measure the actual deployed prediction path of the first-seed one-NTP and
recurrent-NTP models. This is a cost measurement,not another quality-selection
experiment. The future head is removed;no weights,source sequence,objective,
original inference exit or scientific success criterion changes.

Use the original cooled1.005B seed5231 checkpoints. Verify complete serialized
model/optimizer identity before extracting only their prediction weights.
Independent pinned-upstream CPU FP32 references cover64 consecutive natural
TRAIN windows from the already declared mature-probe source position. Preserve
all real-target masks. The same prefix-available position/noise rule and inputs
serve both models. No example is selected for favorable losses or timing.

CPU tests require exact head-removal equivalence and immutable weights/state at
every supported depth,in both FP32 and mixed FP16 execution. The actual-size
CPU reference records full-vocabulary masked likelihood and eight evenly spaced
complete-vocabulary logit vectors per window. Raw source,noise and logits stay
private;publish hashes and small aggregate qualification records.

## Admission and bound

The CPU preparation may overlap MAIN,using zero chips and a300-second cap.
The native queue waits for complete unchanged MAIN and its original NTP
secondary. If NTP qualifies,the four matched-work controls must complete first,
regardless of whether they win. If it fails,require the work queue's terminal
no-admission receipt. A stopped prerequisite does not trigger an automatic
inference recovery. Require no other native owners before admission.

The common preparation/native/postprocessing budget is1,800 seconds. Its
prospective estimate uses four times the measured training microstep plus twice
measured cold training setup. It includes two paired rounds,CPU reference
allowance and reporting. This is a conservative feasibility estimate,not an
inference throughput claim. Subtract actual CPU preparation from the remaining
native/postprocessing clock. Native phases are separately capped at660 seconds.
Resume retains the original deadline and exact saved source list.

Use two independently owned eight-chip workers,32 serving ranks each,local
batch one. Reverse submission order in the second round. Record actual hardware
assignment,and disclose whether each model visited both halves;submission order
alone does not prove hardware balance. Cancellation touches owned workers only.

## Exact measured operation

Execute the existing qualified staged FP16-body,FP32-attention/LM-head path,
with all parameters frozen. The one-pass control executes its supported exit
one. Recurrent weights execute supported exits one,two and three. Exit two
remains the discovery primary. No optimizer,backward,sampling or halting policy.

Each rank alternates two different preloaded original source windows. Before
each path's timing,all ranks must pass both windows against independent CPU
FP32 likelihood (0.003-nat tolerance) and sampled logits (2% relative L2).
Then execute100 warm forwards and100 timed forwards,with device synchronization
after each call. Repeat both numerical checks afterward and verify immutable
weights,zero gradients and allowed fallback operators. Count204 forwards per
rank/depth,including checks. No failed or missing depth may be discarded.

Timed calls return full device-resident vocabulary logits. They exclude
tokenization,host input transfer,CPU output transfer,the numerical loss check,
networking and decoding. Cold compilation,numerical checks,model loading and
phase cleanup remain in separate elapsed/campaign costs. Keep source-window
presentations distinct from unique data:timing repeats64 original windows and
does not train on new targets.

Report latency distributions,rank-duration throughput,both rounds and dense
forward matrix counts including the tied vocabulary projection. Within the same
recurrent weights/hardware,depth two versus one measures the extra pass's cost;
also report the trained one-pass alternative. Rank/timing distributions are
descriptive,not independent training-seed confidence intervals.

These are fixed1024-position **prefills with full logits and preloaded inputs**.
KV-cached autoregressive generation is not implemented or measured. The staged
prototype is not claimed to be an optimized serving implementation. No
peak-FLOP utilization,training-efficiency or terminal-positive claim follows
from this benchmark. Fresh confirmation and strong ordinary controls remain
separate requirements.
